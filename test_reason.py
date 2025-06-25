from transformers import AutoProcessor, AutoModelForImageTextToText, GenerationConfig
from datasets import load_dataset
import torch
from PIL import Image
import json

# === Model and Prompt ===
model_id = "jomoll/gemma-reason1"

system_message = "You are an expert radiologist."
user_prompt = (
    "You are given a chest X-ray image. Please assess different findings on the following scale: "
    "0: none, 1: mild, 2: moderate, 3: severe, 4: very severe. The findings are: "
    "Heart Size, Pulmonary Congestion, Pleural Effusion Right, Pleural Effusion Left, "
    "Pulmonary Opacities Right, Pulmonary Opacities Left, Atelectasis Right, Atelectasis Left.\n\n"
    "Please provide a step-by-step reasoning of your observations from the image first, "
    "and conclude with a final assessment in the following format:\n"
    "{'Heart Size': <value>, ..., 'Atelectasis Left': <value>}."
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


# === Load model and processor ===
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)
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




def format_data(sample):
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



# === Load and Prepare Dataset ===
raw_datasets = load_dataset("jomoll/TAIX-reasoning-v2.1")["val"]

eval_example = format_data(raw_datasets[1])  # pick one for evaluation
eval_messages = eval_example["messages"]


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
        max_new_tokens=2500,
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
