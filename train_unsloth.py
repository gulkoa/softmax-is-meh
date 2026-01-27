"""
Training Script for Adaptive Softmax with Unsloth

Fine-tunes gemma-3n-e2b with:
1. Per-layer trainable polynomial parameters for adaptive temperature softmax
2. LoRA via unsloth for efficient training
3. CLRS algorithmic reasoning dataset (minimum task)

Usage:
    python train_unsloth.py
"""

import os
import sys
import torch
from torch import nn
from pathlib import Path

# Add CLRS repo to path
sys.path.append(os.path.join(os.getcwd(), 'clrs-repo'))

# --- Configuration ---
MODEL_NAME = "google/gemma-3n-e2b"
NUM_LAYERS = 34  # gemma-3n-e2b has 34 transformer layers
MAX_SEQ_LENGTH = 1024
OUTPUT_DIR = "./results_adaptive"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Training hyperparameters
MAX_STEPS = 200
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LR_LORA = 2e-4
LR_POLY = 2e-3  # 10x higher for polynomial params
SAVE_STEPS = 50
LOGGING_STEPS = 10

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 60)
print("Adaptive Softmax Training with Unsloth")
print("=" * 60)

# --- 1. Load Model with Unsloth ---
print("\n[1/6] Loading model with unsloth...")
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,  # Fast 4-bit quantization
    load_in_8bit=False,
)

print(f"  Model: {MODEL_NAME}")
print(f"  Layers: {NUM_LAYERS}")

# --- 2. Apply LoRA ---
print("\n[2/6] Applying LoRA adapters...")
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=8,
    lora_dropout=0,
)

# --- 3. Create Adaptive Polynomial Config ---
print("\n[3/6] Creating per-layer polynomial config...")
from adaptive_softmax import AdaptivePolynomialConfig, softmax_adaptive_temperature

poly_config = AdaptivePolynomialConfig(num_layers=NUM_LAYERS)
poly_config = poly_config.to(model.device)

print(f"  Polynomial parameters: {poly_config.poly_coeffs.shape}")
print(f"  Total trainable poly params: {poly_config.poly_coeffs.numel()}")

# --- 4. Patch Attention with Adaptive Softmax ---
print("\n[4/6] Patching attention layers...")

# Store reference to poly_config globally for the patched softmax
_GLOBAL_POLY_CONFIG = poly_config
_CURRENT_LAYER_IDX = [0]  # Mutable container to track current layer

# Patch torch.nn.functional.softmax to use adaptive version for attention
import torch.nn.functional as F
_original_softmax = F.softmax
_in_attention = [False]

def patched_softmax(input, dim=None, _stacklevel=3, dtype=None):
    """Patched softmax that applies adaptive temperature in attention contexts."""
    global _GLOBAL_POLY_CONFIG, _CURRENT_LAYER_IDX
    
    # Heuristic: attention weights are 4D tensors with softmax on last dim
    if _in_attention[0] and input.dim() == 4 and dim == -1:
        layer_idx = min(_CURRENT_LAYER_IDX[0], NUM_LAYERS - 1)
        coeffs = _GLOBAL_POLY_CONFIG.get_layer_coeffs(layer_idx)
        result = softmax_adaptive_temperature(input, dim, coeffs, dtype=dtype or torch.float32)
        return result.to(input.dtype) if dtype is None else result
    
    # Default behavior for non-attention softmax
    if dtype is not None:
        return _original_softmax(input, dim=dim, dtype=dtype)
    return _original_softmax(input, dim=dim)

F.softmax = patched_softmax

# Wrap each attention layer to track current layer index
def wrap_attention_layer(layer_module, layer_idx):
    original_forward = layer_module.forward
    
    def wrapped_forward(*args, **kwargs):
        _CURRENT_LAYER_IDX[0] = layer_idx
        _in_attention[0] = True
        try:
            result = original_forward(*args, **kwargs)
        finally:
            _in_attention[0] = False
        return result
    
    layer_module.forward = wrapped_forward

# Apply wrapper to all attention layers
# Navigate to the actual transformer layers
if hasattr(model, 'model'):
    base_model = model.model
    if hasattr(base_model, 'model'):
        base_model = base_model.model
    if hasattr(base_model, 'layers'):
        layers = base_model.layers
    elif hasattr(base_model, 'language_model') and hasattr(base_model.language_model, 'model'):
        layers = base_model.language_model.model.layers
    else:
        layers = []
        print("  Warning: Could not find transformer layers, using flat patching")
else:
    layers = []

for idx, layer in enumerate(layers):
    if hasattr(layer, 'self_attn'):
        wrap_attention_layer(layer.self_attn, idx)

print(f"  Wrapped {len(layers)} attention layers")

# --- 5. Setup Dataset (CLRS) ---
print("\n[5/6] Setting up CLRS dataset...")
from datasets import IterableDataset, Features, Value
from clrs._src.clrs_text.huggingface_generators import clrs_generator

algos_and_lengths = {
    "minimum": [8, 16, 24, 32],  # Various lengths for robustness
}

def create_clrs_dataset():
    ds = IterableDataset.from_generator(
        clrs_generator,
        features=Features({
            "text": Value("string"),
            "question": Value("string"),
            "answer": Value("string"),
            "algo_name": Value("string"),
            "length": Value("int32"),
            "use_hints": Value("bool_"),
        }),
        gen_kwargs={
            "algos_and_lengths": algos_and_lengths,
        },
    )
    return ds

dataset = create_clrs_dataset()

# Format for chat template
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

def formatting_func(examples):
    """Format CLRS examples for training."""
    texts = []
    for q, a in zip(examples['question'], examples['answer']):
        # Create chat format
        convo = [
            {"role": "user", "content": f"Find the index of the minimum value.\n{q}"},
            {"role": "assistant", "content": str(a).strip()},
        ]
        text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        if text.startswith('<bos>'):
            text = text[5:]  # Remove leading <bos> as processor adds it
        texts.append(text)
    return {"text": texts}

# Take a finite sample for training
print("  Converting to finite dataset...")
train_samples = []
sample_iter = iter(dataset)
for _ in range(1000):  # 1000 training samples
    try:
        train_samples.append(next(sample_iter))
    except StopIteration:
        break

from datasets import Dataset
train_dataset = Dataset.from_list(train_samples)
train_dataset = train_dataset.map(formatting_func, batched=True, remove_columns=train_dataset.column_names)
print(f"  Training samples: {len(train_dataset)}")

# --- 6. Training ---
print("\n[6/6] Starting training...")
from trl import SFTTrainer, SFTConfig

# Custom callback to save polynomial checkpoints
class PolynomialCheckpointCallback:
    def __init__(self, poly_config, checkpoint_dir, save_steps):
        self.poly_config = poly_config
        self.checkpoint_dir = checkpoint_dir
        self.save_steps = save_steps
        self.step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1
        if self.step % self.save_steps == 0:
            path = os.path.join(self.checkpoint_dir, f"poly_coeffs_step_{self.step}.pt")
            self.poly_config.save_checkpoint(path)
            print(f"\n  Saved polynomial checkpoint: {path}")

# Create optimizer with separate param groups
from torch.optim import AdamW

# Get all trainable model params
model_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW([
    {"params": model_params, "lr": LR_LORA},
    {"params": poly_config.parameters(), "lr": LR_POLY},
])

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=10,
    max_steps=MAX_STEPS,
    learning_rate=LR_LORA,  # This is for the default optimizer, we override
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=5,  # Keep last 5 checkpoints
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=training_args,
    optimizers=(optimizer, None),  # Use our custom optimizer
)

# Manually handle polynomial checkpointing during training
original_training_step = trainer.training_step

def custom_training_step(model, inputs):
    result = original_training_step(model, inputs)
    
    # Save polynomial checkpoint at save_steps
    if trainer.state.global_step > 0 and trainer.state.global_step % SAVE_STEPS == 0:
        poly_path = os.path.join(CHECKPOINT_DIR, f"poly_coeffs_step_{trainer.state.global_step}.pt")
        poly_config.save_checkpoint(poly_path)
        print(f"\n  Saved polynomial checkpoint: {poly_path}")
    
    return result

trainer.training_step = custom_training_step

# Start training
print("\n" + "=" * 60)
print("Training started!")
print("=" * 60)
trainer.train()

# --- Save Final Results ---
print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)

# Save final polynomial coefficients
final_poly_path = os.path.join(OUTPUT_DIR, "poly_coeffs_final.pt")
poly_config.save_checkpoint(final_poly_path)
print(f"\nSaved final polynomial coefficients: {final_poly_path}")

# Save model
model.save_pretrained(os.path.join(OUTPUT_DIR, "model_final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "model_final"))
print(f"Saved model: {os.path.join(OUTPUT_DIR, 'model_final')}")

# Print final coefficients
print("\nFinal polynomial coefficients per layer:")
print("-" * 50)
for i in range(min(NUM_LAYERS, 10)):  # Show first 10 layers
    coeffs = poly_config.poly_coeffs[i].detach().cpu().tolist()
    print(f"  Layer {i:2d}: [{', '.join(f'{c:.4f}' for c in coeffs)}]")
if NUM_LAYERS > 10:
    print(f"  ... ({NUM_LAYERS - 10} more layers)")

print("\nDone!")
