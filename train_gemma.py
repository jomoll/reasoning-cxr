# pip install accelerate
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import torch
import os
import yaml
from datasets import Dataset

model_name = "google/medgemma-4b-it"
image_dir = "data/keno_1000/data_png"
label_dir = "data/keno_1000/annotations/v1.0"
prompt_base = (
            "Analyze this chest X-ray following these steps:\n" 
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
            "Use the exact format:\n" 
            "Reasoning:\n" 
            "  - Step:\n" 
            "    Description: [Step]\n" 
            "    Action:\n" 
            "    - [Observation]\n" 
            "    Result: [Conclusion]"
)

def load_model(model_name=model_name):
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def load_data(image_dir, label_dir, processor):
    dataset = []
    for filename in ["0a000b841142f0763421a9e15f00bd6aff96e70e4c11baddd8ccb27990fc311c.yaml"]:
        if filename.endswith('.yaml'):
            image_id = os.path.splitext(filename)[0]
            image_path = os.path.join(image_dir, f"{image_id}.png")
            if not os.path.exists(image_path):
                continue
                
            # Load image and process
            image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiologist."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_base},
                        {"type": "image", "image": image}
                    ]
                }
            ]

            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

            return inputs

# Load processor & model
model, processor = load_model(model_name)
# Load dataset
inputs = load_data(image_dir, label_dir, processor)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
