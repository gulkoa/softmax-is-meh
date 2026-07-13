"""
Hard-stretch MQAR: AS-Stieltjes arm (flash ASStieltjesMHA — the composition
that beat softmax by up to +30pp on max-retrieval 16k stretch).

Identical data/protocol to zoology_mqar_hardstretch.py (32k vocab, train<=256,
eval to 16k = 64x, early stopping off) so results merge with the sdpa /
flashn-q{4,16} baselines of that sweep. Arms: as-flashn q_order {2, 4} x lr
{1e-3, 3.2e-3}.
"""
import os
import uuid

from zoology.config import (DataConfig, LoggerConfig, ModelConfig,
                            ModuleConfig, TrainConfig)
from zoology.data.multiquery_ar import MQARConfig

sweep_name = "mqar-hardstretch-as-" + uuid.uuid4().hex[:6]

# Env-configurable for robustness sweeps (defaults = original run)
AS_SEEDS = [int(x) for x in os.environ.get("AS_SEEDS", "123").split(",")]
AS_QORDERS = [float(x) for x in os.environ.get("AS_QORDERS", "2,4").split(",")]
AS_LRS = [float(x) for x in os.environ.get("AS_LRS", "1e-3,3.2e-3").split(",")]

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
    batch_size=(256, 4),
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
for sq in AS_QORDERS:
    models.append(ModelConfig(
        block_type="TransformerBlock", d_model=D_MODEL, n_layers=2,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, dict(
                name="stieltjes_mixer.ASStieltjesMHA",
                kwargs={"dropout": 0.0, "num_heads": 2, "stieltjes_q": sq,
                        "num_iter": 8, "normalize": True,
                        "compute_dtype": "bf16"})]},
        ),
        max_position_embeddings=0, name=f"as-flashn-q{sq:g}-d{D_MODEL}",
        **model_factory_kwargs,
    ))

configs = []
for model in models:
    for lr in AS_LRS:
        for seed in AS_SEEDS:
            run_id = f"hs-{model.name}-lr{lr:.1e}-s{seed}"
            configs.append(TrainConfig(
                model=model,
                data=data,
                learning_rate=lr,
                max_epochs=24,
                seed=seed,
                early_stopping_metric="valid/accuracy",
                early_stopping_threshold=1.01,
                logger=LoggerConfig(project_name="stieltjes-flash-attn",
                                    entity="gulkoa"),
                slice_keys=["num_kv_pairs", "input_seq_len"],
                sweep_id=sweep_name,
                run_id=run_id,
            ))
