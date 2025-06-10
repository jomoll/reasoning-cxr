import os
import pandas as pd
import transformers 
import torch
import yaml
from shutil import copyfile

# constants
metadata_path = "data/keno_1000/metadata.csv"
image_dir = "data/keno_1000/data_png"
yaml_output_dir = "data/keno_1000/annotations/v1.0"
template_path = "template_llama.yaml"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
version = str(1.0)

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
    """Clean and format the YAML output for better readability."""
    # Check if input is already a dictionary
    if isinstance(output_text, dict):
        return output_text
        
    # Handle empty or invalid input
    if not output_text or 'Reasoning:' not in output_text:
        return {'Reasoning': []}

    # Initialize the formatted output
    formatted_output = {
        'Reasoning': []
    }

    try:
        # Split into steps and process each
        steps = [s.strip() for s in output_text.split('- Step:') if s.strip()]
        
        for step in steps:
            step_dict = {}
            
            # Get Description
            if 'Description:' in step:
                desc_parts = step.split('Description:', 1)
                desc = desc_parts[1].split('Action:', 1)[0].strip()
                step_dict['Description'] = desc
            
            # Get Actions
            if 'Action:' in step:
                action_parts = step.split('Action:', 1)
                actions = action_parts[1].split('Result:', 1)[0]
                # Clean up action items
                action_items = [item.strip('- ').strip() for item in actions.split('\n') 
                              if item.strip().startswith('-')]
                step_dict['Action'] = action_items
            
            # Get Result
            if 'Result:' in step:
                result = step.split('Result:', 1)[1].split('\n', 1)[0]
                step_dict['Result'] = result.strip().strip('"')
            
            if step_dict:  # Only append if we found some content
                formatted_output['Reasoning'].append(step_dict)
                
        return formatted_output
        
    except Exception as e:
        print(f"Error formatting YAML: {e}")
        return {'Reasoning': []}

model = transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir='.', trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir='.', trust_remote_code=True)

# Load metadata
metadata_df = pd.read_csv(metadata_path)
metadata_df.set_index("UID", inplace=True)

# Define mappings
cardio_map = {-1: "not assessable", 0: "normal", 1: "borderline", 2: "enlarged", 4: "massively enlarged"}
other_map = {0: "none", 1: "mild", 2: "moderate", 3: "severe", 4: "very severe"}

# Create output directory if it doesn't exist
os.makedirs(yaml_output_dir, exist_ok=True)

# Iterate through metadata instead of image directory
for uid, row in metadata_df.iterrows():
    
    image_path = os.path.join(image_dir, f"{uid}.png")
    yaml_output_path = os.path.join(yaml_output_dir, f"{uid}.yaml")
    
    # Check if image exists
    #if not os.path.exists(image_path):
    #    print(f"Skipping {uid}: Image file not found.")
    #    continue
        
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
                max_new_tokens=2048,
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
    except Exception as e:
        print(f"Error processing {uid}: {e}")

    # Stop after processing 1 images for testing
    if uid == metadata_df.index[9]:  # Change this to control how many images to process
        print("Processed 10 images, stopping for testing.")
        break
print(f"Saved annotations to {yaml_output_dir}")