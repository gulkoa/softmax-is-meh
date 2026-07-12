"""
Zoology MQAR: softmax attention vs Stieltjes (FLASHn Triton) — systematic
comparison on the June-2 meeting's designated testbed, without waiting on the
handed-off task code.

Data: the original MQAR recipe (train seq<=256, eval extrapolates to 1024).
Models (2-layer TransformerBlocks, conv+attention hybrid mixers, identical
except the attention op):
  - softmax MHA (zoology.mixers.attention.MHA)         [baseline]
  - StieltjesMHA normalize=True, q in {2, 4}           [our kernel]
Both arms use dropout=0.0 (the fused kernel cannot drop attention probs, so
equal treatment requires it) and the same lr sweep.

Run (login/compute node with GPU):
  PYTHONPATH=/users/PAS2402/alexg/softmax/zoology:/users/PAS2402/alexg/softmax/softmax-is-meh/triton/dev_scripts \
  python -m zoology.launch \
    softmax-is-meh/triton/dev_scripts/zoology_mqar_stieltjes.py

wandb: project stieltjes-flash-attn, entity gulkoa (offline on compute nodes).
"""
import uuid

import numpy as np

from zoology.config import (DataConfig, LoggerConfig, ModelConfig,
                            ModuleConfig, TrainConfig)
from zoology.data.multiquery_ar import MQARConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "mqar-stieltjes-vs-softmax-" + sweep_id

VOCAB_SIZE = 8_192
CACHE_DIR = "/users/PAS2402/alexg/softmax/softmax-is-meh/results/zoology_cache"

train_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=100_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=64),
]
test_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=1_000, num_kv_pairs=128),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256),
]

input_seq_len = max(c.input_seq_len for c in train_configs + test_configs)
batch_size = 256
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(batch_size, batch_size // 8),
    cache_dir=CACHE_DIR,
)

model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}),
    "vocab_size": VOCAB_SIZE,
}

conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={"l_max": input_seq_len, "kernel_size": 3, "implicit_long_conv": True},
)

models = []
for d_model in [64, 128]:
    # --- softmax baseline (dropout 0.0 for parity with the fused kernel) ---
    softmax_mixer = dict(
        name="zoology.mixers.attention.MHA",
        kwargs={"dropout": 0.0, "num_heads": 2},
    )
    models.append(ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=2,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, softmax_mixer]},
        ),
        max_position_embeddings=0,
        name=f"softmax-d{d_model}",
        **model_factory_kwargs,
    ))

    # --- Stieltjes FLASHn arms ---
    for sq in [2.0, 4.0]:
        stj_mixer = dict(
            name="stieltjes_mixer.StieltjesMHA",
            kwargs={"dropout": 0.0, "num_heads": 2,
                    "stieltjes_q": sq, "num_iter": 8, "normalize": True,
                    "compute_dtype": "bf16"},
        )
        models.append(ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=ModuleConfig(
                name="zoology.mixers.hybrid.Hybrid",
                kwargs={"configs": [conv_mixer, stj_mixer]},
            ),
            max_position_embeddings=0,
            name=f"flashn-q{sq:g}-d{d_model}",
            **model_factory_kwargs,
        ))

# Env-trimmable sweep (comma-separated), e.g. SWEEP_DMODELS=64 SWEEP_NLRS=2
import os  # noqa: E402
_dmodels = [int(x) for x in os.environ.get("SWEEP_DMODELS", "64,128").split(",")]
_nlrs = int(os.environ.get("SWEEP_NLRS", "4"))
models = [m for m in models if any(m.name.endswith(f"-d{d}") for d in _dmodels)]

configs = []
for model in models:
    for lr in np.logspace(-3, -1.5, _nlrs):
        run_id = f"{model.name}-lr{lr:.1e}"
        configs.append(TrainConfig(
            model=model,
            data=data,
            learning_rate=lr,
            max_epochs=32,
            logger=LoggerConfig(project_name="stieltjes-flash-attn",
                                entity="gulkoa"),
            slice_keys=["num_kv_pairs"],
            sweep_id=sweep_name,
            run_id=run_id,
        ))
