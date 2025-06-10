import os
import pandas as pd
import transformers 
import torch
import yaml
from shutil import copyfile

# constants
metadata_path = "data/keno_1000/metadata.csv"
image_dir = "data/keno_1000/data_png"
yaml_output_dir = "data/keno_1000/annotations"
template_path = "template_llama.yaml"
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Prompt
with open("example_output.txt", "r") as f:
    example_output = f.read()
prompt_base = (
    "You are a board-certified radiologist tasked with interpreting a chest X-ray image. "
    "Please follow a systematic diagnostic approach to produce a clear, structured report. "
    "Your output must follow this format:\n\n"
    "Reasoning:\n"
    " - Step:\n"
    "   Description: \n"
    "    Action:\n"
    "    - ...\n"
    "    - ...\n"
    "    Result: \n"
    "- Step: ..."
    "Final answer:\n"
    "[Summary diagnosis or clinical impression]\n\n"
    "Ensure that your reasoning is thorough, clinically relevant, and easy to follow. "
    "Use step-by-step anatomical logic, describing where you are looking and what you are seeing. "
    "You are provided with an example below to guide the style and level of detail expected:\n\n"
    + example_output +
    "\nNow, write the report for the clinical diagnosis.\n\n"
)


model = transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir='.', trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir='.', trust_remote_code=True)

# Load metadata
metadata_df = pd.read_csv(metadata_path)
metadata_df.set_index("UID", inplace=True)

# Define mappings
cardio_map = {-1: "not assessable", 0: "normal", 1: "borderline", 2: "enlarged", 4: "massively enlarged"}
other_map = {0: "none", 1: "mild", 2: "moderate", 3: "severe", 4: "very severe"}

def describe_row(row):
    parts = [f"Patient age {int(row['Age'])//365} years"]
    parts.append(f"{cardio_map.get(row['cardiomegaly'], 'unknown')} cardiomegaly")
    parts.append(f"{other_map.get(row['congestion'], 'unknown')} congestion")
    parts.append(f"{other_map.get(row['pleural_effusion_right'], 'unknown')} right pleural effusion")
    parts.append(f"{other_map.get(row['pleural_effusion_left'], 'unknown')} left pleural effusion")
    parts.append(f"{other_map.get(row['pneumonic_infiltrates_right'], 'unknown')} right pneumonic infiltrates")
    parts.append(f"{other_map.get(row['pneumonic_infiltrates_left'], 'unknown')} left pneumonic infiltrates")
    parts.append(f"{other_map.get(row['atelectasis_right'], 'unknown')} right atelectasis")
    parts.append(f"{other_map.get(row['atelectasis_left'], 'unknown')} left atelectasis")
    return "Clinical data: " + ", ".join(parts) + "."

# Create output directory if it doesn't exist
os.makedirs(yaml_output_dir, exist_ok=True)

for filename in os.listdir(image_dir):
    if not filename.lower().endswith(".png"):
        continue

    uid = filename.split(".")[0]
    yaml_output_path = os.path.join(yaml_output_dir, f"{uid}.yaml")

    if uid not in metadata_df.index:
        print(f"Skipping {filename}: UID not found in metadata.")
        continue
    
    try:
        # Copy template file
        copyfile(template_path, yaml_output_path)
        
        row = metadata_df.loc[uid]
        clinical_info = describe_row(row)

        prompt = prompt_base + "\n\n" + clinical_info
        
        # Format prompt for LLaMA
        formatted_prompt = f"[INST] {prompt} [/INST]"
        
        # Tokenize and generate output as before
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True)
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_text = output_text[len(formatted_prompt):].strip()

         # Read existing YAML
        with open(yaml_output_path, 'r') as f:
            yaml_content = yaml.safe_load(f)

        # Update YAML content while preserving structure
        yaml_content['image-type'] = "chest-x-ray"
        yaml_content['version'] = "1.0"
        yaml_content['annotator'] = "Llama-3.3-70B-Instruct"
        yaml_content['image-id'] = uid
        
        # Convert row data to a metadata dictionary
        metadata = row.to_dict()
        # Convert numpy types to native Python types for YAML serialization
        metadata = {k: v.item() if hasattr(v, 'item') else v for k, v in metadata.items()}
        yaml_content['metadata'] = metadata
        yaml_content['reasoning'] = output_text

        # Write updated YAML with proper formatting
        with open(yaml_output_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print(f"Saved annotations to {yaml_output_dir}")