import os
import pandas as pd
import transformers 
import torch
import yaml
from shutil import copyfile
from tqdm import tqdm 
from datasets import load_dataset

# constants
version = str(2.0)
metadata_path = "data/keno_1000/metadata.csv"
yaml_output_dir = "data/keno_1000/annotations/v"+version
template_path = "template_llama.yaml"
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Replace the iteration section
total_images = 10000  # Number of images to process
processed = 0

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
    "FinalAssessment:\n"
    "[Summary diagnosis or clinical impression]\n\n"
    "Ensure that your reasoning is thorough, clinically relevant, and easy to follow. "
    "Use step-by-step anatomical logic, describing where you are looking and what you are seeing. "
    "You are provided with an example below to guide the style and level of detail expected:\n\n"
    + example_output +
    "\nNow, write the report for the clinical diagnosis.\n\n"
)

def describe_row(row):
    parts = [f"Patient age {int(row['Age'])//365} years"]
    parts.append(f"{cardio_map.get(row['HeartSize'], 'unknown')} cardiomegaly")
    parts.append(f"{other_map.get(row['PulmonaryCongestion'], 'unknown')} pulmonary congestion")
    parts.append(f"{other_map.get(row['PleuralEffusion_Right'], 'unknown')} right pleural effusion")
    parts.append(f"{other_map.get(row['PleuralEffusion_Left'], 'unknown')} left pleural effusion")
    parts.append(f"{other_map.get(row['PulmonaryOpacities_Right'], 'unknown')} right pulmonary opacities")
    parts.append(f"{other_map.get(row['PulmonaryOpacities_Left'], 'unknown')} left pulmonary opacities")
    parts.append(f"{other_map.get(row['Atelectasis_Right'], 'unknown')} right atelectasis")
    parts.append(f"{other_map.get(row['Atelectasis_Left'], 'unknown')} left atelectasis")
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

model = transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir='.', trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir='.', trust_remote_code=True)

dataset = load_dataset("TLAIM/TAIX-Ray", name="default")["train"]
metadata_df = pd.DataFrame(dataset)

# Define mappings
cardio_map = {-1: "not assessable", 0: "normal", 1: "borderline", 2: "enlarged", 4: "massively enlarged"}
other_map = {0: "none", 1: "mild", 2: "moderate", 3: "severe", 4: "very severe"}

# Create output directory if it doesn't exist
os.makedirs(yaml_output_dir, exist_ok=True)

# Iterate through metadata file
with tqdm(total=total_images, desc="Writing Reasoning traces...") as pbar:
    # check if uid already in data/annotations/v1.0
    if os.path.exists(yaml_output_dir):
        existing_uids = {os.path.splitext(f)[0] for f in os.listdir(yaml_output_dir) if f.endswith('.yaml')}
    # Iterate through each row in the metadata DataFrame
    for i, row in metadata_df.iterrows():
        uid = row['UID']
        if uid in existing_uids:
            continue
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
                    max_new_tokens=2248,
                    do_sample=True,
                    num_beams=4,
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
            # remove image from dict
            metadata.pop('Image', None)
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