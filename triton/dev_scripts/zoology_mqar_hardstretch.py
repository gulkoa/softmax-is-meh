"""
HARD-STRETCH MQAR: the ASEntmax-style extreme extrapolation protocol, where
softmax and length-adaptive methods actually separate (their paper: softmax 3%
vs ASEntmax 95% at 1024x).

Train: standard MQAR mix (seq <= 256). Test: extrapolation to 65,536 tokens
(256x the longest train seq), kv_pairs scaled ~seq/16.

Arms (all memory-linear so 65k eval is feasible):
  sdpa       : softmax via flash SDPA (stieltjes_mixer.SdpaMHA)
  flashn-qQ  : Triton Stieltjes normalize=True, q in {4, 16}

Trimmed lr sweep {1e-3, 3.2e-3}; early stopping DISABLED (threshold > 1) so
all arms get the full budget — aggregate early stopping masked OOD differences
in the first run.
"""
import os
import uuid

import numpy as np

from zoology.config import (DataConfig, LoggerConfig, ModelConfig,
                            ModuleConfig, TrainConfig)
from zoology.data.multiquery_ar import MQARConfig

sweep_name = "mqar-hardstretch-" + uuid.uuid4().hex[:6]

# zoology's multiquery_ar asserts vocab_size > input_seq_len AND its
# no-replacement sampler materializes a (num_examples x vocab) matrix —
# 131k vocab OOM-killed the host (job 12331838). Cap the stretch at 16k eval
# (64x train) with a 32k vocab: generation peaks ~13GB, still deep in the
# regime where ASEntmax reports softmax collapsing.
VOCAB_SIZE = 32_768
CACHE_DIR = "/users/PAS2402/alexg/softmax/softmax-is-meh/results/zoology_cache"

train_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=100_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=64),
]
test_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=4096, num_examples=500, num_kv_pairs=256),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=8192, num_examples=500, num_kv_pairs=512),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=16384, num_examples=250, num_kv_pairs=1024),
]

input_seq_len = max(c.input_seq_len for c in train_configs + test_configs)
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(256, 4),   # tiny test batch: 65k sequences
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
D_MODEL = 64
# softmax baseline via flash SDPA (memory-linear at 65k)
models.append(ModelConfig(
    block_type="TransformerBlock", d_model=D_MODEL, n_layers=2,
    sequence_mixer=ModuleConfig(
        name="zoology.mixers.hybrid.Hybrid",
        kwargs={"configs": [conv_mixer, dict(
            name="stieltjes_mixer.SdpaMHA",
            kwargs={"dropout": 0.0, "num_heads": 2})]},
    ),
    max_position_embeddings=0, name=f"sdpa-d{D_MODEL}",
    **model_factory_kwargs,
))
for sq in [4.0, 16.0]:
    models.append(ModelConfig(
        block_type="TransformerBlock", d_model=D_MODEL, n_layers=2,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, dict(
                name="stieltjes_mixer.StieltjesMHA",
                kwargs={"dropout": 0.0, "num_heads": 2, "stieltjes_q": sq,
                        "num_iter": 8, "normalize": True,
                        "compute_dtype": "bf16"})]},
        ),
        max_position_embeddings=0, name=f"flashn-q{sq:g}-d{D_MODEL}",
        **model_factory_kwargs,
    ))

configs = []
for model in models:
    for lr in [1e-3, 3.2e-3]:
        run_id = f"hs-{model.name}-lr{lr:.1e}"
        configs.append(TrainConfig(
            model=model,
            data=data,
            learning_rate=lr,
            max_epochs=24,
            early_stopping_metric="valid/accuracy",
            early_stopping_threshold=1.01,   # disabled: run the full budget
            logger=LoggerConfig(project_name="stieltjes-flash-attn",
                                entity="gulkoa"),
            slice_keys=["num_kv_pairs", "input_seq_len"],
            sweep_id=sweep_name,
            run_id=run_id,
        ))
