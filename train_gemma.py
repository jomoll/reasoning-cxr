from datasets import Dataset, load_dataset
from PIL import Image
import os
import yaml
from trl import SFTTrainer
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig

system_message = "You are an expert radiologist."
user_prompt = "You are given a chest X-ray image. Please assess different findings on the following scale: 0: none, 1: mild, 2: moderate, 3: severe, 4: very severe. The findings are: Heart Size, Pulmonary Congestion, Pleural Effusion Right, Pleural Effusion Left, Pulmonary Opacities Right, Pulmonary Opacities Left, Atelectasis Right, Atelectasis Left. Please use the following format for your response: " \
"Heart Size: <value>, Pulmonary Congestion: <value>, Pleural Effusion Right: <value>, Pleural Effusion Left: <value>, Pulmonary Opacities Right: <value>, Pulmonary Opacities Left: <value>, Atelectasis Right: <value>, Atelectasis Left: <value>."

# Convert dataset to OAI messages
def format_data(sample):
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    {
                        "type": "image",
                        "image": sample["Image"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Heart Size:" + str(sample["HeartSize"]), "Pulmonary Congestion": str(sample["PulmonaryCongestion"]), "Pleural Effusion Right": str(sample["PleuralEffusion_Right"]), "Pleural Effusion Left": str(sample["PleuralEffusion_Left"]), "Pulmonary Opacities Right": str(sample["PulmonaryOpacities_Right"]), "Pulmonary Opacities Left": str(sample["PulmonaryOpacities_Left"]), "Atelectasis Right": str(sample["Atelectasis_Right"]), "Atelectasis Left": str(sample["Atelectasis_Left"])}],
            },
        ],
    }

def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs

dataset = load_dataset("TLAIM/TAIX-Ray", name="default")["train"]
dataset = [format_data(sample) for sample in dataset]

# Hugging Face model id
model_id = "google/gemma-3-4b-pt" # or `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`

# Check if GPU benefits from bfloat16
if torch.cuda.get_device_capability()[0] < 8:
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch.bfloat16, # What torch dtype to use, defaults to auto
    device_map="auto", # Let torch decide how to load the model
)

# BitsAndBytesConfig int-4 config
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
)

# Load model and tokenizer
model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)

args = SFTConfig(
    output_dir="gemma-test1",     # directory to save and repository id
    num_train_epochs=1,                         # number of training epochs
    per_device_train_batch_size=1,              # batch size per device during training
    gradient_accumulation_steps=4,              # number of steps before performing a backward/update pass
    gradient_checkpointing=True,                # use gradient checkpointing to save memory
    optim="adamw_torch_fused",                  # use fused adamw optimizer
    logging_steps=5,                            # log every 5 steps
    save_strategy="epoch",                      # save checkpoint every epoch
    learning_rate=2e-4,                         # learning rate, based on QLoRA paper
    bf16=True,                                  # use bfloat16 precision
    max_grad_norm=0.3,                          # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                          # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",               # use constant learning rate scheduler
    push_to_hub=True,                           # push model to hub
    report_to="tensorboard",                    # report metrics to tensorboard
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },  # use reentrant checkpointing
    dataset_text_field="",                      # need a dummy field for collator
    dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
)
args.remove_unused_columns = False # important for collator

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
    labels = batch["input_ids"].clone()

    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch 

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()
