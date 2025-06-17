from transformers import (
    Gemma3ForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os
import yaml
from PIL import Image
import torch

model_name="google/gemma-3-4b-it"
image_dir = "data/keno_1000/data_png"
label_dir = "data/keno_1000/annotations/v1.0" # each label is in a yaml file with the same name as the image, the field is "reasoning"

def load_data(image_dir, label_dir):
    """Load image paths and their corresponding reasoning chains."""
    dataset = []
    
    for filename in os.listdir(label_dir):
        if filename.endswith('.yaml'):
            image_id = os.path.splitext(filename)[0]
            image_path = os.path.join(image_dir, f"{image_id}.png")
            
            if not os.path.exists(image_path):
                continue
                
            with open(os.path.join(label_dir, filename), 'r') as f:
                label_data = yaml.safe_load(f)
            
            # Create messages format
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text", 
                            "text": "You are a board-certified radiologist analyzing chest X-rays."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": image_path  # Local file path
                        },
                        {
                            "type": "text",
                            "text": "Analyze this chest X-ray following these steps:\n" + 
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
                                    "Use the exact format:\n" +
                                    "Reasoning:\n" +
                                    "  - Step:\n" +
                                    "    Description: [Step]\n" +
                                    "    Action:\n" +
                                    "    - [Observation]\n" +
                                    "    Result: [Conclusion]"
                        }
                    ]
                }
            ]
            
            # Format reasoning chain as completion
            reasoning = label_data.get('reasoning', {}).get('Reasoning', [])
            completion = ""
            for step in reasoning:                    
                description = step.get('Description', '')
                if description:
                    completion += f"Description: {description}\n"
                
                actions = step.get('Action', [])
                if actions:
                    completion += "Actions:\n"
                    for action in actions:
                        completion += f"- {action}\n"
                
                result = step.get('Result', '')
                if result:
                    completion += f"Result: {result}\n"
                
                completion += "\n"
            
            dataset.append({
                'messages': messages,
                'completion': completion
            })
    # Add this before return Dataset.from_list(dataset)
    if not dataset:
        print("Warning: No valid data was loaded!")
    else:
        print(f"Successfully loaded {len(dataset)} samples")
        # Print first sample for verification
        print("\nFirst sample:")
        print("Messages:", dataset[0]['messages'])
        print("Completion:", dataset[0]['completion'][:100], "...")
    return Dataset.from_list(dataset)

def main():
    # Initialize model and tokenizer
    model = Gemma3ForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.float16,
        device_map="auto"
        )
    processor = AutoProcessor.from_pretrained(model_name)

    # Load dataset
    dataset = load_data(image_dir=image_dir, 
                       label_dir=label_dir)
    inputs = processor.apply_chat_template(
        dataset.messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)

if __name__ == "__main__":
    main()

