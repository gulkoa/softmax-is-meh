import os
import sys
# print("Importing torch...")
import torch
from torch import nn
# print("Importing transformers...")
import transformers
# print("Importing peft...")
from peft import LoraConfig, get_peft_model, TaskType
# print("Importing datasets...")
from datasets import IterableDataset, Features, Value
import numpy as np

# Add CLRS repo to path
sys.path.append(os.path.join(os.getcwd(), 'clrs-repo'))
print("Importing clrs...")
from clrs._src.clrs_text.huggingface_generators import clrs_generator
print("Imports done.")

# --- 1. Adaptive Softmax Logic ---

# Initial coefficients (from paper)
# We make this a Parameter so it can be learned
POLY_FIT = torch.nn.Parameter(torch.tensor([-0.037, 0.481, -2.3, 4.917, -1.791]), requires_grad=True)

def get_polynomial_value(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    cur_val = torch.zeros_like(x)
    # c is [a4, a3, a2, a1, a0] corresponding to x^4, x^3...
    # Horner's method or simple loop
    for i in range(len(c) - 1):
        cur_val = (cur_val + c[i]) * x
    return cur_val + c[-1]

def softmax_adaptive_temperature(logits, dim, poly_fit, dtype=torch.float32):
    """
    Adaptive temperature softmax.
    """
    original_probs = torch.softmax(logits, dim=dim, dtype=dtype)
    # Compute Shannon entropy
    # Add small epsilon to avoid log(0)
    entropy = torch.sum(-original_probs * torch.log(original_probs + 1e-9), dim=-1, keepdim=True)
    
    # Calculate beta (inverse temperature)
    # We use the poly_fit parameter passed in
    poly_val = get_polynomial_value(entropy, poly_fit)
    
    # Apply guards
    # 1. Low entropy guard: if H < 0.5, beta = 1.0
    # 2. Dispersion guard: beta >= 1.0 (never increase entropy)
    beta = torch.where(
        entropy > 0.5,
        torch.maximum(poly_val, torch.tensor(1.0, device=logits.device, dtype=logits.dtype)),
        torch.tensor(1.0, device=logits.device, dtype=logits.dtype)
    )
    
    return torch.softmax(logits * beta, dim=dim, dtype=dtype)

def adaptive_temperature_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor = None,
    dropout: float = 0.0,
    scaling: float = None,
    softcap: float = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    if scaling is None:
        scaling = module.head_dim**-0.5

    # Access the learnable parameter
    poly_fit_param = getattr(module, 'poly_fit', POLY_FIT)

    # Standard Gemma attention logic
    key_states = transformers.models.gemma3.modeling_gemma3.repeat_kv(key, module.num_key_value_groups)
    value_states = transformers.models.gemma3.modeling_gemma3.repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # --- ADAPTIVE SOFTMAX ---
    attn_weights = softmax_adaptive_temperature(attn_weights, dim=-1, poly_fit=poly_fit_param, dtype=torch.float32).to(query.dtype)
    # ------------------------

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    
    return attn_output, attn_weights

# Register the function
transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS.register("adaptive_temperature_eager", adaptive_temperature_eager_attention_forward)


# --- 2. Model Setup ---

MODEL_NAME = "google/gemma-3-1b-it"

print("Loading model...")
model = transformers.Gemma3ForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="adaptive_temperature_eager",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Register POLY_FIT to all attention layers so it's part of the model
print("Registering polynomial parameters...")
for name, module in model.named_modules():
    if "layers" in name and "self_attn" in name:
        # Register as a parameter of the module
        # We share the same tensor across all layers
        module.register_parameter('poly_fit', POLY_FIT)

# LoRA Setup
print("Setting up LoRA...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)

# IMPORTANT: Unfreeze the polynomial parameter!
# get_peft_model freezes all non-LoRA params.
for name, param in model.named_parameters():
    if "poly_fit" in name:
        param.requires_grad = True

model.print_trainable_parameters()


# --- 3. Dataset Setup (CLRS) ---

print("Setting up dataset...")
algos_and_lengths = {
    "minimum": [16, 32] # Train on slightly longer sequences to encourage generalization? Or stick to short?
    # Paper says they trained on simple task. Let's use reasonable lengths.
}

def generate_dataset():
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

dataset = generate_dataset()
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def collate_fn(examples):
    # Prepare inputs for the model
    # We want to train it to answer the question.
    # Format: "Question: <q>\nAnswer: <a>"
    texts = []
    for ex in examples:
        # Construct a prompt. 
        # The CLRS generator outputs 'text' which might be the full trace, but 'question' and 'answer' are what we need.
        # Let's use a standard chat format or simple completion.
        # Gemma-it expects chat format usually, but for base completion:
        prompt = f"User: {ex['question']}\nModel: {ex['answer']}"
        texts.append(prompt)
    
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    encodings["labels"] = encodings["input_ids"].clone()
    return encodings

# --- 4. Training ---

print("Starting training...")
training_args = transformers.TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=100, # Short run for demonstration
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=50,
    fp16=True,
    remove_unused_columns=False, # Important for IterableDataset
    report_to="none"
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

trainer.train()

print("Training complete.")
print("Final Polynomial Coefficients:")
print(POLY_FIT.data)

# Save the model (LoRA adapters + the poly_fit param?)
# LoRA save_pretrained only saves adapters.
# We need to manually save the poly_fit if we want to reuse it.
torch.save(POLY_FIT, "./results/poly_fit.pt")
model.save_pretrained("./results/final_model")
