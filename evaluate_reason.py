# === Evaluation Script: Final Assessment Only ===
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, GenerationConfig
import re
import ast
import json
from tqdm import tqdm

# === Constants ===
model_path = "jomoll/gemma-reason1"  # Path to your fine-tuned model
dataset_id = "jomoll/TAIX-reasoning-v2.1"
processor_id = "google/gemma-3-4b-it"
max_new_tokens = 2248

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

# === Load Model and Processor ===
model = AutoModelForImageTextToText.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(processor_id)

# === Load Validation Dataset ===
dataset = load_dataset(dataset_id, split="val")

# === Evaluation Logic ===
def extract_final_assessment(output_text):
    try:
        match = re.search(r"Final assessment:\n?(\{.*\})", output_text)
        if match:
            return ast.literal_eval(match.group(1))
    except Exception:
        return None
    return None

def format_data(sample):
    reasoning_data = json.loads(sample["Reasoning"])
    reasoning_text = "Reasoning:\n"
    for i, step in enumerate(reasoning_data):
        s = step["Step"]
        reasoning_text += f"Step {i+1}:\nDescription: {s.get('Description', '')}\nAction:\n"
        for a in s.get("Action", []):
            reasoning_text += f"- {a}\n"
        reasoning_text += f"Result: {s.get('Result', '')}\n\n"
    final_labels = str({finding: sample[finding] for finding in FINDINGS})
    assistant_response = f"{reasoning_text}\n--- END OF REASONING ---\n\nFinal assessment:\n{final_labels}"
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "You are given a chest X-ray image. Please assess different findings on the following scale: "
                        "0: none, 1: mild, 2: moderate, 3: severe, 4: very severe. The findings are: "
                        "Heart Size, Pulmonary Congestion, Pleural Effusion Right, Pleural Effusion Left, "
                        "Pulmonary Opacities Right, Pulmonary Opacities Left, Atelectasis Right, Atelectasis Left.\n\n"
                        "Please provide a step-by-step reasoning of your observations from the image first, "
                        "and conclude with a final assessment in the following format:\n"
                        "{'Heart Size': <value>, ..., 'Atelectasis Left': <value>}."
                    )},
                    {"type": "image", "image": sample["Image"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]},
        ]
    }

# === Run Evaluation ===
y_true = {f: [] for f in FINDINGS}
y_pred = {f: [] for f in FINDINGS}

for sample in tqdm(dataset, desc="Evaluating"):
    formatted = format_data(sample)
    messages = formatted["messages"]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=5,
            generation_config=GenerationConfig(pad_token_id=processor.tokenizer.pad_token_id)
        )
        output = processor.decode(generation[0][input_len:], skip_special_tokens=True)

    pred_dict = extract_final_assessment(output)
    gt_dict = {finding: sample[finding] for finding in FINDINGS}

    for finding in FINDINGS:
        y_true[finding].append(gt_dict[finding])
        y_pred[finding].append(pred_dict.get(finding, -999) if pred_dict else -999)

# === Compute Accuracy ===
print("\n📊 Evaluation Metrics:")
accs = []
for finding in FINDINGS:
    correct = sum(int(p == t) for p, t in zip(y_pred[finding], y_true[finding]))
    total = len(y_true[finding])
    acc = correct / total
    accs.append(acc)
    print(f"{finding}: {acc:.3f}")
print(f"\n🔢 Average Accuracy: {sum(accs)/len(accs):.3f}")
