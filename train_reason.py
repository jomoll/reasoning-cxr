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


# === Constants & Model Config ===
model_id = "google/gemma-3-4b-pt"
processor_id = "google/gemma-3-4b-it"
dataset_id = "jomoll/TAIX-reasoning-v2.1"

system_message = "You are an expert radiologist."
user_prompt = (
    "You are given a chest X-ray image. Please assess different findings on the following scales:\n"
    "- Heart Size: 0 = normal, 1 = borderline, 2 = enlarged, 4 = massively enlarged\n"
    "- All others: 0 = none, 1 = mild, 2 = moderate, 3 = severe, 4 = very severe\n\n"
    "The findings to report are:\n"
    "  • Heart Size\n"
    "  • Pulmonary Congestion\n"
    "  • Pleural Effusion Right\n"
    "  • Pleural Effusion Left\n"
    "  • Pulmonary Opacities Right\n"
    "  • Pulmonary Opacities Left\n"
    "  • Atelectasis Right\n"
    "  • Atelectasis Left\n\n"
    "First, provide a step-by-step reasoning under the header “Reasoning:” using this exact template for each step:\n"
    "Reasoning:\n"
    "  - Step 1:\n"
    "      Description: <brief description>\n"
    "      Action:\n"
    "        - <what you looked at>\n"
    "        - <what you concluded>\n"
    "      Result: <what you found>\n"
    "  - Step 2: …\n"
    "  …\n"
    "  - Step N: Formulate a final assessment.\n\n"
    "After your last reasoning step, include exactly this sentinel line (no extra text):\n"
    "```text\n"
    "--- END OF REASONING ---\n"
    "Final assessment:\n"
    "``` \n"
    "Then output only the final assessment JSON, e.g.:\n"
    "{'Heart Size': 2, 'Pulmonary Congestion': 1, … 'Atelectasis Left': 0}\n\n"
    "Once you print that JSON, immediately stop and do not generate any further text."
)

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
def format_labels_json(sample):
    return str({finding: sample[finding] for finding in FINDINGS})


def format_reasoning(reasoning_steps):
    lines = ["Reasoning:"]
    for i, step in enumerate(reasoning_steps):
        s = step["Step"]
        lines.append(f"Step {i+1}:")
        lines.append(f"Description: {s.get('Description', '')}")
        lines.append("Action:")
        for a in s.get("Action", []):
            lines.append(f"- {a}")
        lines.append(f"Result: {s.get('Result', '')}\n")
    return "\n".join(lines)




def format_data_train(sample):
    reasoning_data = json.loads(sample["Reasoning"])
    reasoning_text = format_reasoning(reasoning_data)    
    final_labels = format_labels_json(sample)
    assistant_response = f"{reasoning_text}\n\n--- END OF REASONING ---\n\nFinal assessment:\n{final_labels}"

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

def format_data_val(sample):
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


def process_vision_info(messages):
    image_inputs = []
    for msg in messages:
        for item in msg.get("content", []):
            if isinstance(item, dict) and item.get("type") == "image":
                image = item["image"]
                image_inputs.append(image.convert("RGB"))
    return image_inputs


# === Load and Prepare Dataset ===
raw_datasets = load_dataset("jomoll/TAIX-reasoning-v2.1-cleaned")
train_raw = raw_datasets["train"]
val_raw = raw_datasets["val"]

# Limit the number of samples for quick testing
NUM_SAMPLES = 3
val_raw = val_raw.select(range(NUM_SAMPLES))

train_dataset = [format_data_train(sample) for sample in train_raw]
eval_dataset = [format_data_val(sample) for sample in val_raw]
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
    output_dir="gemma-reason10",
    num_train_epochs=10,
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
    save_strategy="epoch",
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
        # Directly extract the assistant's text from the message content
        assistant_text = ex["messages"][-1]["content"][0]["text"]
        
        # Find where the assistant message starts in the full sequence
        full_text = texts[i]
        assistant_start = full_text.find(assistant_text)
        if assistant_start != -1:
            # Tokenize the prefix to find start position
            prefix_text = full_text[:assistant_start]
            prefix_tokens = processor.tokenizer(prefix_text, add_special_tokens=False)
            start_idx = len(prefix_tokens.input_ids)
            
            # Tokenize assistant text to find length
            assistant_tokens = processor.tokenizer(assistant_text, add_special_tokens=False)
            end_idx = start_idx + len(assistant_tokens.input_ids)
            
            # Make sure we don't go beyond sequence length
            if end_idx > labels.shape[1]:
                end_idx = labels.shape[1]
                
            # Only unmask the assistant tokens
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
trainer.train()
trainer.save_model()
print("✅ Training complete and model saved.")

# === Evaluate on Held-Out Sample ===
print("\n🔍 Running evaluation on a single sample from the val split...")

# Reformat eval message (reuse same logic as training)
eval_messages = eval_dataset[0]["messages"]

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
        max_new_tokens=2248,
        do_sample=False,
        num_beams=5,
        generation_config=GenerationConfig(pad_token_id=processor.tokenizer.pad_token_id)
    )
    generation = generation[0][input_len:]

# Decode output
decoded = processor.decode(generation, skip_special_tokens=True)

print("\n🧠 Model Prediction:\n")
print(decoded)

# Print ground truth for comparison
print("\n📌 Ground Truth:")
ground_truth = eval_example["messages"][-1]["content"][0]["text"]
print(ground_truth)
