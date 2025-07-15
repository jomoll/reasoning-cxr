import os
import pandas as pd
import transformers 
import torch
import yaml
from shutil import copyfile
from tqdm import tqdm 

# constants
version = str(3.0)
metadata_path = "data/keno_1000/Metadata_1000_only_new.csv"
yaml_output_dir = "data/keno_1000/annotations/v"+version
template_path = "template_llama.yaml"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Replace the iteration section
total_images = 10000  # Number of images to process
processed = 0

# Prompt
with open("example_output2.txt", "r") as f:
    example_output = f.read()
prompt_base = (
    "You are a board-certified radiologist interpreting a chest X-ray. "
    "Follow a systematic, step-by-step anatomical approach and produce a structured report.\n\n"
    "Your output **must** follow this exact format:\n\n"
    "Reasoning:\n"
    " - Step<number>:\n"
    "   Description: <what you're examining (e.g., 'Cardiac Size and Contours')>\n"
    "   Action: <list of where you look, what you see, and what that means (e.g., '- I assess the cardiothoracic ratio; the heart appears enlarged.')>\n"
    "     - <action 1>\n"
    "     - <action 2>\n"
    "   Result: <intermediate conclusion>\n"
    " - Step...: ...\n\n"
    "Guidelines:\n"
    "1. Use clear, concise anatomical language (“left lower lobe,” “mediastinal contour,” etc.).\n"
    "2. Each step should flow logically: describe **where**, then **what**, then **so what**.\n"
    "3. Keep Actions and Observations bulleted for readability.\n\n"
    "4. Do **not** simply repeat phrases from the example; it’s for reference only.\n\n"
    "Here's an example of a complete trace for style and level of detail:\n\n"
    f"{example_output}\n\n"
    "Now, given the following clinical case, write the Reasoning:\n\n"
)

def describe_row(row):
    parts = [f"Patient age {int(row['Age'])//365} years"]
    parts.append(f"{cardio_map.get(row['cardiomegaly2'], 'unknown')} cardiomegaly")
    parts.append(f"{other_map.get(row['congestion2'], 'unknown')} congestion")
    parts.append(f"{other_map.get(row['pleural_effusion_right2'], 'unknown')} right pleural effusion")
    parts.append(f"{other_map.get(row['pleural_effusion_left2'], 'unknown')} left pleural effusion")
    parts.append(f"{other_map.get(row['pneumonic_infiltrates_right2'], 'unknown')} right pneumonic infiltrates")
    parts.append(f"{other_map.get(row['pneumonic_infiltrates_left2'], 'unknown')} left pneumonic infiltrates")
    parts.append(f"{other_map.get(row['atelectasis_right2'], 'unknown')} right atelectasis")
    parts.append(f"{other_map.get(row['atelectasis_left2'], 'unknown')} left atelectasis")
    parts.append(f"{pneumo_map.get(row['pneumothorax_right'], 'unknown')} right pneumothorax")
    parts.append(f"{pneumo_map.get(row['pneumothorax_left'], 'unknown')} left pneumothorax")
    parts.append(f"{(row['Sonstiges'], 'unknown')}")
    return "Clinical data: " + ", ".join(parts) + "."

def clean_yaml_format(output_text):
    if isinstance(output_text, dict):
        return output_text
    if not output_text or 'Reasoning:' not in output_text:
        return {'Reasoning': []}
    # formatted output: 
    # Reasoning: [{- Step 1: Description: ..., Action: [...], Result: ...}, ...]
    # FinalAssessment: [summary diagnosis]
    formatted_output = {'Reasoning': []}
    try:
        steps = [s.strip() for s in output_text.split('- Step') if s.strip()]

        for step in steps:
            step_index = step.split(':')[0].strip()
            step_dict = {}
            if 'Description:' in step:
                desc = step.split('Description:', 1)[1].split('Action:', 1)[0].strip()
                step_dict['Description'] = desc
            if 'Action:' in step:
                actions = step.split('Action:', 1)[1].split('Result:', 1)[0]
                action_items = [item.strip('- ').strip() for item in actions.split('\n') if item.strip().startswith('-')]
                step_dict['Action'] = action_items
            if 'Result:' in step:
                result = step.split('Result:', 1)[1].split('\n', 1)[0]
                step_dict['Result'] = result.strip().strip('"')
            if step_dict:
                formatted_output['Reasoning'].append({f'Step {step_index}': step_dict})  # Changed from {'- Step': step_dict}
        return formatted_output
        
    except Exception as e:
        print(f"Error formatting YAML: {e}")
        return {'Reasoning': []}

model = transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir='.', trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir='.', trust_remote_code=True)

# Load metadata
metadata_df = pd.read_csv(metadata_path)
metadata_df.set_index("UID", inplace=True)

# Add this after loading metadata_df but before the iteration loop
target_uids_file = "target_uids.txt"
if os.path.exists(target_uids_file):
    with open(target_uids_file, 'r') as f:
        target_uids = {line.strip() for line in f if line.strip()}
    # Filter metadata to only include target UIDs
    metadata_df = metadata_df[metadata_df.index.isin(target_uids)]
    print(f"Found {len(metadata_df)} samples matching target UIDs")
else:
    print("No target_uids.txt found, will process all samples")

# Update total_images to not exceed available samples
total_images = min(total_images, len(metadata_df))

# Define mappings
cardio_map = {-1: "not assessable", 0: "normal", 1: "borderline", 2: "enlarged", 4: "massively enlarged"}
other_map = {0: "none", 1: "mild", 2: "moderate", 3: "severe", 4: "very severe"}
pneumo_map = {0: "no", 1: "there is a"}
# Create output directory if it doesn't exist
os.makedirs(yaml_output_dir, exist_ok=True)

# Iterate through metadata file
with tqdm(total=total_images, desc="Writing Reasoning traces...") as pbar:
    # check if uid already in data/annotations/v1.0
    if os.path.exists(yaml_output_dir):
        existing_uids = {os.path.splitext(f)[0] for f in os.listdir(yaml_output_dir) if f.endswith('.yaml')}
        metadata_df = metadata_df[~metadata_df.index.isin(existing_uids)]
    # Iterate through each row in the metadata DataFrame
    for uid, row in metadata_df.iterrows():
        yaml_output_path = os.path.join(yaml_output_dir, f"{uid}.yaml")
            
        try:
            # Copy template file
            copyfile(template_path, yaml_output_path)
            
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
                    max_new_tokens=1800,
                    do_sample=True,
                    num_beams=1,
                    temperature=0.5,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = output_text[len(formatted_prompt):].strip()
            output_text = clean_yaml_format(output_text)  # Add formatting cleanup

            # Read existing YAML
            with open(yaml_output_path, 'r') as f:
                yaml_content = yaml.safe_load(f)

            # Convert row data to a metadata dictionary
            metadata = row.to_dict()
            # Convert numpy types to native Python types for YAML serialization
            metadata = {k: v.item() if hasattr(v, 'item') else v for k, v in metadata.items()}

            # Update YAML content while preserving structure
            yaml_content['image-type'] = "chest-x-ray"
            yaml_content['version'] = version
            yaml_content['annotator'] = model_name
            yaml_content['image-id'] = uid
            yaml_content['metadata'] = metadata
            
            # Format the reasoning section
            formatted_reasoning = {
                'reasoning': clean_yaml_format(output_text)
            }
            
            # Write updated YAML with proper formatting
            with open(yaml_output_path, 'w') as f:
                yaml.dump(
                    {**yaml_content, **formatted_reasoning},
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    width=float("inf"),  # Prevent line wrapping
                    indent=2
                )
            pbar.update(1)
            processed += 1
            
            # Stop after processing desired number of images
            if processed >= total_images:
                break
        except Exception as e:
            pbar.write(f"Error processing {uid}: {e}")
            continue
     
print(f"\nSaved {processed} annotations to {yaml_output_dir}")