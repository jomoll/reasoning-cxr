# === Imports ===
import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    GenerationConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import json
import random


# === Constants & Model Config ===
model_id = "google/medgemma-4b-it"
resume_from_checkpoint = False
path_to_checkpoint = "gemma-reasontest" if resume_from_checkpoint else None
processor_id = "google/medgemma-4b-it"
dataset_id = "jomoll/TAIX-reasoning-v2.1-cleaned-stepwise-filtered"

system_message = "You are an expert radiologist."


FINDINGS = [
    "HeartSize",
    "PulmonaryCongestion",
    "PleuralEffusion_Right",
    "PleuralEffusion_Left",
    "PulmonaryOpacities_Right",
    "PulmonaryOpacities_Left",
    "Atelectasis_Right",
    "Atelectasis_Left",
]

# === Device & Precision Check ===
if torch.cuda.get_device_capability()[0] < 8:
    raise ValueError("GPU does not support bfloat16, please use one that does.")

# === Model & Processor Loading ===
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    ),
)

model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
processor = AutoProcessor.from_pretrained(processor_id)
print(f"✅ Model `{model_id}` and processor `{processor_id}` loaded.")


# === Data Formatting Functions ===

def format_data(sample):
    reasoning_data = sample["Reasoning"]
    user_prompt = reasoning_data[0]["Step"]["Description"]
    actions = reasoning_data[0]["Step"]["Action"]
    result = reasoning_data[0]["Step"]["Result"]
    formatted_actions = "\n".join([f"• {action}" for action in actions])
    assistant_response = f"<think>\n{formatted_actions}\n</think>\n\n<result>{result}</result>"

    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": sample["Image"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]},
        ]
    }

def process_vision_info(messages):
    image_inputs = []
    for msg in messages:
        for item in msg.get("content", []):
            if isinstance(item, dict) and item.get("type") == "image":
                image = item["image"]
                image_inputs.append(image.convert("RGB"))
    return image_inputs


# === Load and Prepare Dataset ===
raw_datasets = load_dataset(dataset_id)
train_raw = raw_datasets["train"]
train_raw = train_raw.select(range(1000))  # Limit to 1 sample for quick testing
train_raw = train_raw.shuffle(seed=42)
val_raw = raw_datasets["val"]

# Limit the number of samples for quick testing
NUM_SAMPLES = 1
val_raw = val_raw.select(range(NUM_SAMPLES))
train_dataset = [format_data(sample) for sample in train_raw]
eval_dataset = [format_data(sample) for sample in val_raw]

print(f"📊 Training dataset size: {len(train_dataset)} sample(s)")
print(f"📊 Evaluation dataset size: {len(eval_dataset)} sample(s)")


# === PEFT Configuration ===
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"],
)

# === Training Configuration ===
args = SFTConfig(
    output_dir="medgemma-2",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    learning_rate=2e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="no",
    push_to_hub=True,
    report_to="wandb",
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
)


# === Data Collator ===
def collate_fn(examples):
    texts, images = [], []
    for ex in examples:
        image_inputs = process_vision_info(ex["messages"])
        text = processor.apply_chat_template(ex["messages"], add_generation_prompt=False, tokenize=False)
        texts.append(text.strip())
        images.append(image_inputs)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    
    # Mask all tokens by default
    labels.fill_(-100)
    
    for i, ex in enumerate(examples):
        # Only process examples with assistant messages (training examples)
        if len(ex["messages"]) > 2:              
            # Find the special token that marks the start of model/assistant turn
            assistant_markers = [
                "<think>",                # Our custom format
                "<start_of_turn>model",   # Original marker
                "<assistant>"             # Common alternative
            ]            
            full_text = texts[i]
            start_idx = None
            for marker in assistant_markers:
                marker_pos = full_text.find(marker)
                if marker_pos != -1:
                    # Find the actual token position
                    prefix = full_text[:marker_pos]
                    prefix_ids = processor.tokenizer(prefix, add_special_tokens=False).input_ids
                    start_idx = len(prefix_ids)
                    break
            
            if start_idx is not None:
                # Only unmask the assistant tokens
                end_idx = min(batch["input_ids"].shape[1], batch["attention_mask"][i].sum().item())
                labels[i, start_idx:end_idx] = batch["input_ids"][i, start_idx:end_idx]
    
    # Still mask special tokens within the assistant response
    image_token_id = processor.tokenizer.convert_tokens_to_ids(
        processor.tokenizer.special_tokens_map.get("boi_token", "<|image|>")
    )
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100  # Fallback for image token
    batch["labels"] = labels

    return batch


# === Trainer ===
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

# === Train & Save ===
print("🚀 Starting training...")
if resume_from_checkpoint:
    print("🔄 Resuming from checkpoint...")
    trainer.train(resume_from_checkpoint=path_to_checkpoint)
else:
    print("📚 Starting fresh training...")
    trainer.train()
trainer.save_model()
print("✅ Training complete and model saved.")

# === Evaluate on Held-Out Sample ===
print("\n🔍 Running evaluation on a single sample from the val split...")
def format_data_val(sample):
    reasoning_data = sample["Reasoning"]
    user_prompt = reasoning_data[0]["Step"]["Description"]
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": sample["Image"]},
                ],
            }
        ]
    }

test_sample = val_raw[0]
test_dataset = [format_data_val(test_sample)]

# Reformat eval message (reuse same logic as training)
eval_messages = test_dataset[0]["messages"]

# Tokenize with generation prompt
inputs = processor.apply_chat_template(
    eval_messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

# Track position where generation starts
input_len = inputs["input_ids"].shape[-1]
print("🔢 Prompt tokens:", inputs["input_ids"].shape[-1])

# Generate output
with torch.inference_mode():
    generation = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        num_beams=5,
        generation_config=GenerationConfig(pad_token_id=processor.tokenizer.pad_token_id)
    )
    generation = generation[0][input_len:]

# Decode output
decoded = processor.decode(generation, skip_special_tokens=True)

print("\n🧠 Model Prediction:\n")
print(decoded)

# print ground truth for comparison
print("📜 Ground Truth:\n")
ground_truth = test_sample["Reasoning"][0]["Step"]["Action"]+test_sample["Reasoning"][0]["Step"]["Result"]
print(ground_truth)
