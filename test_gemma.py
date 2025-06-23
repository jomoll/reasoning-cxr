from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
import torch
from PIL import Image

# Set your model ID
model_id = "jomoll/gemma-test1"

# Prompts used during training
system_message = "You are an expert radiologist."
user_prompt = (
    "You are given a chest X-ray image. Please assess different findings on the following scale: "
    "0: none, 1: mild, 2: moderate, 3: severe, 4: very severe. The findings are: "
    "Heart Size, Pulmonary Congestion, Pleural Effusion Right, Pleural Effusion Left, "
    "Pulmonary Opacities Right, Pulmonary Opacities Left, Atelectasis Right, Atelectasis Left. "
    "Please use the following format for your response: "
    "Heart Size: <value>, Pulmonary Congestion: <value>, Pleural Effusion Right: <value>, "
    "Pleural Effusion Left: <value>, Pulmonary Opacities Right: <value>, "
    "Pulmonary Opacities Left: <value>, Atelectasis Right: <value>, Atelectasis Left: <value>."
)

# Load model and processor
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# Load the merged dataset with Reasoning traces
dataset = load_dataset("TLAIM/TAIX-Ray", name="default")["train"]
sample = dataset[0]  # use first example
image = sample["Image"].convert("RGB")

# Construct messages in OAI format
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_message}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image", "image": image}
        ]
    }
]

# Tokenize inputs
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

# Generate response
with torch.inference_mode():
    generation = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        num_beams=5
    )
    generation = generation[0][input_len:]

# Decode
decoded = processor.decode(generation, skip_special_tokens=True)
print("Predicted Output:")
print(decoded)

# Optional: show ground truth
#print("\nGround Truth (Reasoning-based label):")
#print(sample["Reasoning"])
