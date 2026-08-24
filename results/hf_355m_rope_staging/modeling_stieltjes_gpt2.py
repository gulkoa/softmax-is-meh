"""Stieltjes-attention GPT-2-style causal LM (research artifact)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_stieltjes_gpt2 import StieltjesGPT2Config


def stieltjes_probs(scores, q_order, iters=50, eps=1e-20):
    """Row-normalized Stieltjes attention over fp32 scores (last dim)."""
    K = scores.shape[-1]
    with torch.no_grad():
        smax = scores.amax(dim=-1, keepdim=True)
        lo = smax + 1e-6
        hi = smax + float(K) ** (1.0 / q_order)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            f = (mid - scores).clamp_min(eps).pow(-q_order).sum(-1, keepdim=True)
            gt = f > 1.0
            lo = torch.where(gt, mid, lo)
            hi = torch.where(gt, hi, mid)
        lam = 0.5 * (lo + hi)
    diff = (lam - scores).clamp_min(eps)
    f_val = diff.pow(-q_order).sum(-1, keepdim=True) - 1.0
    f_der = (-q_order) * diff.pow(-q_order - 1.0).sum(-1, keepdim=True)
    lam = lam - f_val / f_der
    w = (lam - scores).clamp_min(eps).pow(-q_order)
    return w / w.sum(-1, keepdim=True).clamp_min(eps)


class Attn(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.h = config.n_head
        self.hd = config.n_embd // config.n_head
        self.stj_q = config.stj_q
        self.iters = config.bisect_iters
        self.attn_type = getattr(config, "attn", "stieltjes")
        self.layer_idx = layer_idx
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        if getattr(config, "scale_learnable", False):
            self.scale_mult = nn.Parameter(torch.zeros(config.n_head))
        self.rope = getattr(config, "rope", False)
        self.rope_theta = float(getattr(config, "rope_theta", 10000.0))

    def _rope_apply(self, x, positions):
        # rotate-half rotary, fp32 (mirrors the trainer); positions are
        # ABSOLUTE token indices so cached keys stay consistent.
        d = self.hd
        inv = 1.0 / (self.rope_theta ** (
            torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
        f = torch.outer(positions.to(torch.float32), inv)
        cos = torch.cat((f.cos(), f.cos()), dim=-1)[None, None]
        sin = torch.cat((f.sin(), f.sin()), dim=-1)[None, None]
        xf = x.float()
        rot = torch.cat((-xf[..., d // 2:], xf[..., : d // 2]), dim=-1)
        return (xf * cos + rot * sin).to(x.dtype)

    def forward(self, x, past_key_values=None, attention_mask=None):
        B, S, E = x.shape
        q, k, v = self.qkv(x).split(E, dim=2)
        q = q.view(B, S, self.h, self.hd).transpose(1, 2)
        k = k.view(B, S, self.h, self.hd).transpose(1, 2)
        v = v.view(B, S, self.h, self.hd).transpose(1, 2)
        if self.rope:
            past_len = (past_key_values.get_seq_length(self.layer_idx)
                        if past_key_values is not None else 0)
            pos = torch.arange(past_len, past_len + S, device=x.device)
            q = self._rope_apply(q, pos)
            k = self._rope_apply(k, pos)
        if hasattr(self, "scale_mult"):          # tanh-capped, C=15 (mirror trainer)
            eff = 1.0 + 15.0 * torch.tanh(self.scale_mult / 15.0)
            q = q * eff[None, :, None, None].to(q.dtype)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        S_kv = k.shape[2]

        scores = torch.einsum("bhsd,bhtd->bhst", q.float(), k.float())
        scores = scores / math.sqrt(self.hd)
        if S > 1:
            # prefill / training: causal mask over the aligned suffix
            qpos = torch.arange(S_kv - S, S_kv, device=x.device)
            kpos = torch.arange(S_kv, device=x.device)
            mask = qpos[:, None] >= kpos[None, :]
            scores = scores.masked_fill(~mask[None, None], -1e9)
        if attention_mask is not None:
            key_mask = attention_mask[:, :S_kv].to(torch.bool)
            scores = scores.masked_fill(~key_mask[:, None, None, :], -1e9)
        if self.attn_type == "sdpa":
            p = torch.softmax(scores, dim=-1)   # softmax-baseline export
        else:
            p = stieltjes_probs(scores, self.stj_q, self.iters)
        o = torch.einsum("bhst,bhtd->bhsd", p.to(v.dtype), v)
        return self.proj(o.transpose(1, 2).reshape(B, S, E))


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = Attn(config, layer_idx)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False))

    def forward(self, x, past_key_values=None, attention_mask=None):
        x = x + self.attn(self.ln1(x), past_key_values, attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class StieltjesGPT2ForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = StieltjesGPT2Config
    _tied_weights_keys = {"head.weight": "wte.weight"}
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.nope = (getattr(config, "nope", False)
                     or getattr(config, "rope", False))
        if not self.nope:
            self.wpe = nn.Embedding(config.ctx, config.n_embd)
        self.blocks = nn.ModuleList(
            Block(config, i) for i in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, value):
        self.head = value

    def forward(self, input_ids=None, past_key_values=None, labels=None,
                use_cache=None, attention_mask=None, **kwargs):
        B, S = input_ids.shape
        if use_cache and past_key_values is None:
            try:
                past_key_values = DynamicCache(config=self.config)
            except TypeError:      # transformers < 5: no config kwarg
                past_key_values = DynamicCache()
        past_len = (past_key_values.get_seq_length()
                    if isinstance(past_key_values, Cache) else 0)
        x = self.wte(input_ids)
        if not self.nope:                    # positional models add wpe (clamped);
            pos = torch.arange(past_len, past_len + S,  # nope adds nothing, so it
                               device=input_ids.device)  # accepts arbitrary length
            x = x + self.wpe(pos.clamp(max=self.config.ctx - 1))[None]
        for blk in self.blocks:
            x = blk(x, past_key_values if use_cache else None, attention_mask)
        logits = self.head(self.ln_f(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=past_key_values if use_cache else None)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, **kwargs):
        if isinstance(past_key_values, Cache) and \
                past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, past_key_values.get_seq_length():]
        return {"input_ids": input_ids, "past_key_values": past_key_values,
                "attention_mask": attention_mask, "use_cache": True}
