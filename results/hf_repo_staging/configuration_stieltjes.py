"""Configuration for the Stieltjes-attention causal LM."""
from transformers import PretrainedConfig


class StieltjesConfig(PretrainedConfig):
    model_type = "stieltjes_lm"
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "hidden_size": "d_model",
        "num_attention_heads": "n_head",
    }

    def __init__(self, vocab_size=260, n_layer=4, d_model=512, n_head=16,
                 head_dim=32, ffn=1024, stj_q=4.0, bisect_iters=50,
                 nape=True, adaptive_scale=True, use_cache=False,
                 bos_token_id=257, eos_token_id=259, pad_token_id=256,
                 tie_word_embeddings=True, **kwargs):
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = head_dim
        self.ffn = ffn
        self.stj_q = stj_q
        self.bisect_iters = bisect_iters
        self.nape = nape
        self.adaptive_scale = adaptive_scale
        super().__init__(bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id,
                         pad_token_id=pad_token_id,
                         tie_word_embeddings=tie_word_embeddings, **kwargs)
