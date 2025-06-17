import os
import pandas as pd
import yaml
from shutil import copyfile
from tqdm import tqdm 

from openai import AzureOpenAI
import numpy as np

# constants
version = str(1.3)
metadata_path = "data/keno_1000/metadata.csv"
image_dir = "data/keno_1000/data_png"
yaml_output_dir = "data/keno_1000/annotations/v"+version
template_path = "template_llama.yaml"
model_name = "jb-turbo-2024-04-09"  


client = AzureOpenAI(
    api_key="e849b8c4c4a04d3d817aa67d66189251",
    api_version="2024-02-01",
    azure_endpoint="https://jb-turbo-2024-04-09.openai.azure.com/",
)


# Replace the iteration section
total_images = 10  # Number of images to process
processed = 0

# Prompt
with open("example_output.txt", "r") as f:
    example_output = f.read()

prompt_base = (
    "You are a board-certified radiologist tasked with interpreting a chest X-ray image. "
    "You MUST follow these exact 16 steps in order:\n"
    "1. Assess the image quality.\n"
    "2. Look for central venous catheter placement.\n"
    "3. Look for endotracheal tube placement.\n"
    "4. Look for nasogastric tube placement.\n"
    "5. Look for chest tube placement.\n"
    "6. Look for pacemaker placement.\n"
    "7. Look for other devices.\n"
    "8. Look for the heart size.\n"
    "9. Look for mediastinal size and shift.\n"
    "10. Look for cardiac congestion.\n"
    "11. Look for pleural effusion.\n"
    "12. Look for pulmonary atelectasis.\n"
    "13. Look for pneumonic infiltrates.\n"
    "14. Look for pneumothorax.\n"
    "15. Look for soft tissue pathologies.\n"
    "16. Formulate final assessment.\n\n"
    "For each step, use this exact format:\n"
    "Reasoning:\n"
    "  - Step:\n"
    "    Description: [Step name from above]\n"
    "    Action:\n"
    "    - [Specific observation]\n"
    "    Result: [Conclusion]\n\n"
    "Below is a reference example showing the exact structure to follow:\n\n"
    f"{example_output}\n"
    "Now, write the report following these exact steps and format.\n\n"
)

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

def clean_yaml_format(output_text):
    if isinstance(output_text, dict):
        return output_text
    if not output_text or 'Reasoning:' not in output_text:
        return {'Reasoning': []}
    # formatted output: 
    # Reasoning: [{- Step: Description: ..., Action: [...], Result: ...}, ...]
    # FinalAssessment: [summary diagnosis]
    formatted_output = {'Reasoning': []}
    try:
        steps = [s.strip() for s in output_text.split('- Step:') if s.strip()]
        for step in steps:
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
                formatted_output['Reasoning'].append({'Step': step_dict})  # Changed from {'- Step': step_dict}

        # Check for FinalAssessment
        if 'FinalAssessment:' in output_text:
            final_assessment = output_text.split('FinalAssessment:', 1)[1].strip()
            # Format as a list if not already
            if not final_assessment.startswith('- '):
                final_assessment = [final_assessment.strip().strip('"')]
            else:
                # Split into list items if multiple are present
                final_assessment = [item.strip().strip('- ').strip('"') 
                                 for item in final_assessment.split('\n') 
                                 if item.strip()]
            formatted_output['FinalAssessment'] = final_assessment
        else:
            formatted_output['FinalAssessment'] = []
        
        return formatted_output
    
    except Exception as e:
        print(f"Error formatting YAML: {e}")
        return {'Reasoning': []}

# Load metadata
metadata_df = pd.read_csv(metadata_path)
metadata_df.set_index("UID", inplace=True)

# Optional: filter based on target_uids.txt if exists
target_uids_file = "target_uids.txt"
if os.path.exists(target_uids_file):
    with open(target_uids_file, 'r') as f:
        target_uids = {line.strip() for line in f if line.strip()}
    metadata_df = metadata_df[metadata_df.index.isin(target_uids)]
    print(f"Found {len(metadata_df)} samples matching target UIDs")
else:
    print("No target_uids.txt found, will process all samples")

total_images = min(total_images, len(metadata_df))

# Define mappings
cardio_map = {-1: "not assessable", 0: "normal", 1: "borderline", 2: "enlarged", 4: "massively enlarged"}
other_map = {0: "none", 1: "mild", 2: "moderate", 3: "severe", 4: "very severe"}

os.makedirs(yaml_output_dir, exist_ok=True)

# Main loop
with tqdm(total=total_images, desc="Writing Reasoning traces...") as pbar:
    existing_uids = set()
    if os.path.exists(yaml_output_dir):
        existing_uids = {os.path.splitext(f)[0] for f in os.listdir(yaml_output_dir) if f.endswith('.yaml')}
        metadata_df = metadata_df[~metadata_df.index.isin(existing_uids)]

    for uid, row in metadata_df.iterrows():
        image_path = os.path.join(image_dir, f"{uid}.png")
        yaml_output_path = os.path.join(yaml_output_dir, f"{uid}.yaml")

        try:
            # Copy template file
            copyfile(template_path, yaml_output_path)

            clinical_info = describe_row(row)
            full_prompt = prompt_base + "\n\n" + clinical_info

            # Send request to Azure OpenAI
            response = client.chat.completions.create(
                model=model_name,
                temperature=0,
                n=1,
                max_tokens=2500,
                messages=[
                    {"role": "system", "content": "You are a board-certified radiologist producing reasoning chains for chest X-ray interpretation."},
                    {"role": "user", "content": full_prompt}
                ]
            )

            output_text = response.choices[0].message.content.strip()
            output_text = clean_yaml_format(output_text)

            # Read existing YAML
            with open(yaml_output_path, 'r') as f:
                yaml_content = yaml.safe_load(f)

            metadata_dict = row.to_dict()
            metadata_dict = {k: v.item() if hasattr(v, 'item') else v for k, v in metadata_dict.items()}

            yaml_content['image-type'] = "chest-x-ray"
            yaml_content['version'] = version
            yaml_content['annotator'] = model_name
            yaml_content['image-id'] = uid
            yaml_content['metadata'] = metadata_dict

            formatted_reasoning = {'reasoning': clean_yaml_format(output_text)}

            with open(yaml_output_path, 'w') as f:
                yaml.dump({**yaml_content, **formatted_reasoning}, f,
                          default_flow_style=False, sort_keys=False, allow_unicode=True, width=float("inf"), indent=2)

            pbar.update(1)
            processed += 1

            if processed >= total_images:
                break
        except Exception as e:
            pbar.write(f"Error processing {uid}: {e}")
            continue

print(f"\nSaved {processed} annotations to {yaml_output_dir}")
