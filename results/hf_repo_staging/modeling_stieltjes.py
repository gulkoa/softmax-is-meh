"""Stieltjes-attention causal LM (research artifact)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from .configuration_stieltjes import StieltjesConfig


def stieltjes_probs(scores, q_order, iters=50, eps=1e-20):
    """Row-normalized Stieltjes attention over fp32 scores."""
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


class StieltjesAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.h = config.n_head
        self.hd = config.head_dim
        self.stj_q = config.stj_q
        self.iters = config.bisect_iters
        self.nape = config.nape
        inner = self.h * self.hd
        self.qkv = nn.Linear(config.d_model, 3 * inner, bias=False)
        self.proj = nn.Linear(inner, config.d_model, bias=False)
        self.n_scaled = self.h // 2 if config.nape else self.h
        # slopes are computed in forward(): non-persistent buffers are not
        # restored under meta-device init
        if config.adaptive_scale:
            self.beta_proj = nn.Linear(config.d_model, self.n_scaled, bias=True)
            self.gamma_proj = nn.Linear(config.d_model, self.n_scaled, bias=True)
            self.gamma_range = 2.0
            self.delta = 1.0

    def _scale(self, x, S):
        beta = F.softplus(self.beta_proj(x.float())).transpose(1, 2)
        gamma = (self.gamma_range
                 * torch.tanh(self.gamma_proj(x.float()))).transpose(1, 2)
        logn = torch.log(torch.arange(2, S + 2, device=x.device,
                                      dtype=torch.float32))
        return self.delta + beta * logn.view(1, 1, -1).pow(gamma)

    def forward(self, x, attention_mask=None):
        B, S, _ = x.shape
        q, k, v = self.qkv(x).split(self.h * self.hd, dim=2)
        q = q.view(B, S, self.h, self.hd).transpose(1, 2)
        k = k.view(B, S, self.h, self.hd).transpose(1, 2)
        v = v.view(B, S, self.h, self.hd).transpose(1, 2)

        if hasattr(self, "beta_proj"):
            scale = self._scale(x, S)                      # (B, n_scaled, S)
            if self.n_scaled < self.h:
                ones = torch.ones(B, self.h - self.n_scaled, S,
                                  device=x.device, dtype=scale.dtype)
                scale = torch.cat([ones, scale], dim=1)
            q = q.float() * scale.unsqueeze(-1)

        scores = torch.einsum("bhsd,bhtd->bhst", q.float(), k.float())
        scores = scores / math.sqrt(self.hd)
        if self.nape:
            slopes = torch.cat([
                1.0 / torch.arange(1, self.h // 2 + 1, device=x.device,
                                   dtype=torch.float32),
                torch.zeros(self.h - self.h // 2, device=x.device)])
            pos = torch.arange(S, device=x.device)
            alibi = torch.tril(pos[None, :] - pos[:, None]).float()
            scores = scores + slopes.view(1, -1, 1, 1) * alibi
        causal = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, -1e9)
        if attention_mask is not None:                     # mask PAD keys
            key_mask = attention_mask[:, None, None, :].to(torch.bool)
            scores = scores.masked_fill(~key_mask, -1e9)

        p = stieltjes_probs(scores, self.stj_q, self.iters)
        o = torch.einsum("bhst,bhtd->bhsd", p.to(v.dtype), v)
        o = o.transpose(1, 2).reshape(B, S, self.h * self.hd)
        return self.proj(o)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = StieltjesAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.ffn, bias=False), nn.GELU(),
            nn.Linear(config.ffn, config.d_model, bias=False))

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class StieltjesForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = StieltjesConfig
    _tied_weights_keys = {"head.weight": "emb.weight"}
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False

    def __init__(self, config):
        super().__init__(config)
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config)
                                    for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.emb

    def set_input_embeddings(self, value):
        self.emb = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, value):
        self.head = value

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                **kwargs):
        x = self.emb(input_ids)
        for blk in self.blocks:
            x = blk(x, attention_mask)
        logits = self.head(self.ln_f(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None,
                                      **kwargs):
        # no KV cache: re-run the full sequence each step
        return {"input_ids": input_ids, "attention_mask": attention_mask}
