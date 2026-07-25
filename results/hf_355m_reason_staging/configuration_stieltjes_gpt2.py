"""Configuration for the Stieltjes-attention GPT-2-style LM."""
from transformers import PretrainedConfig


class StieltjesGPT2Config(PretrainedConfig):
    model_type = "stieltjes_gpt2"
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "max_position_embeddings": "ctx",
    }

    def __init__(self, vocab_size=50304, n_layer=12, n_head=12, n_embd=768,
                 ctx=1024, stj_q=4.0, bisect_iters=50, use_cache=True,
                 attn="stieltjes", bos_token_id=50256, eos_token_id=50256,
                 tie_word_embeddings=True, **kwargs):
        self.attn = attn                     # "stieltjes" | "sdpa"
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.ctx = ctx
        self.stj_q = stj_q
        self.bisect_iters = bisect_iters
        self.use_cache = use_cache
        super().__init__(bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id,
                         tie_word_embeddings=tie_word_embeddings, **kwargs)
