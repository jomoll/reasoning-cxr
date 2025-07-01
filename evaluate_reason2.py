# === Eval Script: Final Assessment Only ===
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, GenerationConfig
import re, ast
from tqdm import tqdm
import json
import os

# --- Constants ---
model_id       = "jomoll/gemma-reasontest"   
processor_id   = "google/medgemma-4b-it"
dataset_id     = "jomoll/TAIX-reasoning-v2.1-cleaned-stepwise-filtered"
output_dir     = "results"
max_new_tokens = 2300
NUM_SAMPLES    = 10
NUM_BEAMS      = 5  
system_message = "You are an expert radiologist."

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


# --- Load model + processor ---
model     = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(processor_id)
print(f"✅ Model `{model_id}` and processor `{processor_id}`loaded.")
model.eval()

# --- Load validation split ---
val_dataset_raw = load_dataset(dataset_id, split="val")
# only use the first x samples for quick testing
val_dataset_raw = val_dataset_raw.select(range(NUM_SAMPLES)) 
print(f"📊 Validation dataset size: {len(val_dataset_raw)} sample(s)")


# --- Run evaluation ---
y_true = {c: [] for c in FINDINGS}
y_pred = {c: [] for c in FINDINGS}
results = []

for sample in tqdm(val_dataset_raw, desc="🚀 Starting evaluation..."):
    uid = sample["UID"]
    ref = str(sample["Reasoning"][0]["Step"]["Action"]) + str(sample["Reasoning"][0]["Step"]["Result"])
    # 0) format data
    sample_formatted = format_data_val(sample)
    eval_messages = sample_formatted["messages"]
    # 1) raw prompt string
    inputs = processor.apply_chat_template(
        eval_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    # 4) generate
    input_len = inputs["input_ids"].size(1)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=NUM_BEAMS,
            generation_config=GenerationConfig(
                pad_token_id=processor.tokenizer.pad_token_id
            )
        )
    generated = out[0, input_len:]

    # 7) decode & extract
    text_out = processor.decode(generated, skip_special_tokens=True)

    # save refs and predictions for comparison to text file
    with open("compare_results.txt", "a") as f:
        f.write(f"UID: {uid}\n")
        f.write(f"Ground Truth: {ref}\n")
        f.write(f"Model Prediction: {text_out}\n\n")
    """
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
# append per category accuracies to results
cat_accs = {cat: f"{acc:.3f}" for cat, acc in zip(FINDINGS, accs)}
print(f"\n🔢 Average Accuracy: {sum(accs)/len(accs):.3f}")
print("\n✅ Evaluation complete!")

# --- Save everything ---
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, f"{model_id.replace('/', '_')}_eval_results.json")
print(f"💾 Saving detailed outputs to {out_path}...")
with open(out_path, "w") as fp:
    json.dump({
        "per_category_accuracy": f"{cat_accs}",
        "average_accuracy": f"{sum(accs)/len(accs):.3f}",
        "detailed_results": results
    }, fp, indent=2)
print(f"✅ Detailed outputs saved to {out_path}")
"""