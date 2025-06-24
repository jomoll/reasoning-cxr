# === Eval Script: Final Assessment Only ===
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, GenerationConfig
import re, ast
from tqdm import tqdm
import json

# --- Constants ---
model_id       = "jomoll/gemma-reason2"    # your fine-tuned model
dataset_id     = "jomoll/TAIX-reasoning-v2.1"
max_new_tokens = 2248
system_message = "You are an expert radiologist."

user_prompt = (
    "You are given a chest X-ray image. Please assess different findings on the following scales:\n"
    "- Heart Size: 0 = normal, 1 = borderline, 2 = enlarged, 3 = severely enlarged, 4 = massively enlarged\n"
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
    "HeartSize","PulmonaryCongestion","PleuralEffusion_Right","PleuralEffusion_Left",
    "PulmonaryOpacities_Right","PulmonaryOpacities_Left","Atelectasis_Right","Atelectasis_Left",
]

# --- Helpers ---
def extract_final_assessment(text):
    m = re.search(r"Final assessment:\s*(\{.*\})", text, flags=re.DOTALL)
    if not m: return {}
    try:
        # unify quotes then parse
        return ast.literal_eval(m.group(1))
    except:
        return {}

def process_vision_info(messages):
    image_inputs = []
    for msg in messages:
        for item in msg.get("content", []):
            if isinstance(item, dict) and item.get("type") == "image":
                image = item["image"]
                image_inputs.append(image.convert("RGB"))
    return image_inputs

# --- Load model + processor ---
model     = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)
print(f"✅ Model `{model_id}` and processor loaded.")
model.eval()

# --- Load validation split ---
val_dataset = load_dataset(dataset_id, split="val")
# only use the first x samples for quick testing
val_dataset = val_dataset.select(range(3)) 
print(f"📊 Validation dataset size: {len(val_dataset)} sample(s)")

# --- Run evaluation ---
y_true = {c: [] for c in FINDINGS}
y_pred = {c: [] for c in FINDINGS}
results = []

for sample in tqdm(val_dataset, desc="🚀 Starting evaluation..."):
    uid = sample["UID"]
    # 1) format the chat messages
    messages = [
        {"role":"system", "content":[{"type":"text","text":system_message}]},
        {"role":"user",   "content":[
            {"type":"text","text":user_prompt},
            {"type":"image","image":sample["Image"].convert("RGB")},
        ]},
    ]

    # 2) extract the image list exactly like collate_fn
    images = process_vision_info(messages)  # returns [PIL.Image]

    # 3) build the text prompt, but do NOT tokenize yet
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    ).strip()

    # 4) now call the processor on BOTH modalities
    inputs = processor(
        text=[text],        # list of one prompt
        images=[images],    # list of one image‐list
        return_tensors="pt",
        padding=True
    ).to(model.device, dtype=torch.bfloat16)

    # 5) measure prompt length
    input_len = inputs["input_ids"].size(1)

    # 6) generate
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=5,
            generation_config=GenerationConfig(
                pad_token_id=processor.tokenizer.pad_token_id
            )
        )
    generated = out[0, input_len:]  # shape: [new_tokens]

    # 7) decode & extract
    text_out = processor.decode(generated, skip_special_tokens=True)
    pred = extract_final_assessment(text_out)

    results.append({
        "UID": uid,
        "ground_truth_assessment": {k: sample[k] for k in FINDINGS},
        "predicted_assessment": pred,
        "full_output": text_out,
    })
    # 8) record predictions
    for cat in FINDINGS:
        y_true[cat].append(sample[cat])
        y_pred[cat].append(pred.get(cat, -999))

# --- Compute & print accuracies ---
print("\n📊 Per-Category Accuracies:")
accs = []
for cat in FINDINGS:
    correct = sum(int(p==t) for p,t in zip(y_pred[cat], y_true[cat]))
    total   = len(y_true[cat])
    acc      = correct/total
    accs.append(acc)
    print(f"{cat:25s}: {acc:.3f}")

print(f"\n🔢 Average Accuracy: {sum(accs)/len(accs):.3f}")
print("\n✅ Evaluation complete!")
with open(f"{model_id}.json") as f:
    json.dump(results, f, indent=2)

print("✅ Detailed outputs saved to eval_results.json")