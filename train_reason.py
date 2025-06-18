from datasets import Dataset, load_dataset
from PIL import Image
import os
import yaml
from trl import SFTTrainer
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig
# Hugging Face model id
model_id = "google/gemma-3-4b-pt" # or `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`
system_message = "You are an expert radiologist."
user_prompt = "You are given a chest X-ray image. Please assess different findings on the following scale: 0: none, 1: mild, 2: moderate, 3: severe, 4: very severe. The findings are: Heart Size, Pulmonary Congestion, Pleural Effusion Right, Pleural Effusion Left, Pulmonary Opacities Right, Pulmonary Opacities Left, Atelectasis Right, Atelectasis Left. Please use the following format for your response: " \
"Heart Size: <value>, Pulmonary Congestion: <value>, Pleural Effusion Right: <value>, Pleural Effusion Left: <value>, Pulmonary Opacities Right: <value>, Pulmonary Opacities Left: <value>, Atelectasis Right: <value>, Atelectasis Left: <value>."
reasoning_sample = """Reasoning:
  - Step:
      Description: Assess the image quality.
    Action:
      - I am looking at the chest X-ray image.
      - The image appears to be clear and well-defined.
      - There are no significant artifacts or obstructions.
      - There is no significant rotation of the image.
      - The lungs are fully visible.
      - The patient is positioned symmetrically.
    Result: "The image is of diagnostic quality."
  - Step:
    Description: Look for central venous catheter placement.
    Action:
      - I am looking for the presence of a central venous catheter in the chest X-ray.
      - I am looking at the right side of the neck for a central venous catheter.
      - I do not see a central venous catheter at the right side of the neck.
      - I am looking at the left side of the neck for a central venous catheter.
      - I do not see a central venous catheter at the left side of the neck.
      - I am looking at the right subvlavian vein for a central venous catheter.
      - I do not see a central venous catheter at the right subclavian vein.
      - I am looking at the left subvlavian vein for a central venous catheter.
      - I do not see a central venous catheter at the left subclavian vein.
    Result: "No central venous catheter is present in the chest X-ray."
  - Step:
    Description: Look for endotracheal tube placement.
    Action:
      - I am looking for the presence of an endotracheal tube in the chest X-ray.
      - I am looking at the trachea for an endotracheal tube.
      - I do not see an endotracheal tube in the trachea.
    Result: "No endotracheal tube is present in the chest X-ray."
  - Step:
    Description: Look for nasogastric tube placement.
    Action:
      - I am looking for the presence of a nasogastric tube in the chest X-ray.
      - I am looking at the esophagus for a nasogastric tube.
      - I do not see a nasogastric tube in the esophagus.
    Result: "No nasogastric tube is present in the chest X-ray."
  - Step:
    Description: Look for chest tube placement.
    Action:
      - I am looking for the presence of a chest tube in the chest X-ray.
      - I am looking at the right side of the chest for a chest tube.
      - I do not see a chest tube at the right side of the chest.
      - I am looking at the left side of the chest for a chest tube.
      - I do not see a chest tube at the left side of the chest.
    Result: "No chest tube is present in the chest X-ray."
  - Step:
    Description: Look for pacemaker placement.
    Action:
      - I am looking for the presence of a pacemaker in the chest X-ray.
      - I am looking at the right side of the chest for a pacemaker.
      - I do not see a pacemaker at the right side of the chest.
      - I am looking at the left side of the chest for a pacemaker.
      - I do not see a pacemaker at the left side of the chest.
    Result: "No pacemaker is present in the chest X-ray."
  - Step:
    Description: Look for other devices.
    Action:
      - I am looking for the presence of other devices in the chest X-ray.
      - I am looking for orthopedic hardware.
      - I do not see orthopedic hardware in the chest X-ray.
      - I am looking for any other medical devices.
      - I do not see any other medical devices in the chest X-ray.
    Result: "No other medical devices are present in the chest X-ray."
  - Step:
    Description: Look for the heart size.
    Action:
      - I am looking at the heart size in the chest X-ray.
      - The heart size appears to be normal.
      - The heart contours are clear and well-defined.
    Result: "The heart size is normal."
  - Step:
    Description: Look for mediastinal size and shift.
    Action:
      - I am looking at the mediastinal size and shift in the chest X-ray.
      - The mediastinum appears to be normal in size.
      - The medastinum appears to be midline.
      - The mediastinal contours are clear and well-defined.
    Result: "The mediastinal size is normal and midline."
  - Step:
    Description: Look for cardiac congestion.
    Action:
      - I am looking for signs of cardiac congestion in the chest X-ray.
      - I am looking for pulmonary vascular congestion.
      - There is no Redistribution of pulmonary vascular markings.
      - There are no Kerley B lines.
      - There is no peribronchial cuffing.
      - There is no sign of pulmonary edema.
    Result: "There is no cardiac congestion."
  - Step: Look for pleural effusion.
    Action:
      - I am looking for signs of pleural effusion in the chest X-ray.
      - I am looking at the right side of the chest for pleural effusion.
      - The costophrenic angle on the right side is clear.
      - There is no opacity in the right costophrenic angle.
      - There is no pleural effusion on the right lower side of the chest.
      - I am looking at the left side of the chest for pleural effusion.
      - The costophrenic angle on the left side is clear.
      - There is no opacity in the left lower side of the chest.
      - There is no pleural effusion on the left side of the chest.
    Result: "There is no pleural effusion."
  - Step: Look for pulmonary atelectasis.
    Action:
      - I am looking for signs of pulmonary atelectasis in the chest X-ray.
      - I am looking at the right lung for atelectasis.
      - There is no opacity in the right lung.
      - The right lung appears to be fully expanded.
      - I am looking at the left lung for atelectasis.
      - There is no opacity in the left lung.
      - The left lung appears to be fully expanded.
    Result: "There is no pulmonary atelectasis."
  - Step: Look for pulmonary infiltrates.
    Action:
      - I am looking for signs of pulmonary infiltrates in the chest X-ray.
      - I am looking at the right lung for infiltrates.
      - There is no opacity in the right lung.
      - The right lung appears to be clear.
      - I am looking at the left lung for infiltrates.
      - There is no opacity in the left lung.
      - The left lung appears to be clear.
    Result: "There are no pulmonary infiltrates."
  - Step: Look for pneumothorax.
    Action:
      - I am looking for signs of pneumothorax in the chest X-ray.
      - I am looking at the right lung for pneumothorax.
      - There is no visible pleural line on the right side.
      - There is no evidence of pneumothorax on the right side.
      - I am looking at the left lung for pneumothorax.
      - There is no visible pleural line on the left side.
      - There is no evidence of pneumothorax on the left side.
      - There is no mediastinal shift.
    Result: "There is no pneumothorax."
  - Step: Look for pathologies of the soft tissues.
    Action:
      - I am looking for pathologies of the soft tissues in the chest X-ray.
      - I am looking at the right side of the chest for soft tissue pathologies.
      - There are no abnormalities in the right side of the chest.
      - I am looking at the left side of the chest for soft tissue pathologies.
      - There are no abnormalities in the left side of the chest.
    Result: "There are no pathologies of the soft tissues."
  - Step: Formulate a final assessment.
    Action:
      - I am summarizing the findings from the chest X-ray.
      - The image is of diagnostic quality.
      - No central venous catheter is present.
      - No endotracheal tube is present.
      - No nasogastric tube is present.
      - No chest tube is present.
      - No pacemaker is present.
      - No other medical devices are present.
      - The heart size is normal.
      - The mediastinal size is normal and midline.
      - There is no cardiac congestion.
      - There is no pleural effusion.
      - There is no pulmonary atelectasis.
      - There are no pulmonary infiltrates.
      - There is no pneumothorax.
      - There are no pathologies of the soft tissues.
    Result: "The chest X-ray shows no significant abnormalities."
FinalAssessment:
  - "There are no therapeutic devices present in the chest X-ray."
  - "The chest X-ray is normal with no significant abnormalities detected."
"""


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
print(f"Model {model_id} loaded successfully.")

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
                "content": [
                    {
                        "type": "text", 
                        "text": reasoning_sample,
                    }]
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
dataset = dataset.select(range(1))
dataset = [format_data(sample) for sample in dataset]
print(f"Dataset size: {len(dataset)}")

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
    output_dir="reason-test1",     # directory to save and repository id
    num_train_epochs=100,                         # number of training epochs
    per_device_train_batch_size=1,              # batch size per device during training
    gradient_accumulation_steps=4,              # number of steps before performing a backward/update pass
    gradient_checkpointing=True,                # use gradient checkpointing to save memory
    optim="adamw_torch_fused",                  # use fused adamw optimizer
    logging_steps=5,                            # log every 5 steps
    save_strategy="no",                      # save checkpoint every epoch
    learning_rate=2e-4,                         # learning rate, based on QLoRA paper
    bf16=True,                                  # use bfloat16 precision
    max_grad_norm=0.3,                          # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                          # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",               # use constant learning rate scheduler
    push_to_hub=True,                           # push model to hub
    report_to="wandb",                    # report metrics to tensorboard
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
