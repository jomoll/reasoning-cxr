# pip install accelerate
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import torch

model_id = "jomoll/gemma-test1"
system_message = "You are an expert radiologist."
user_prompt = "You are given a chest X-ray image. Please assess different findings on the following scale: 0: none, 1: mild, 2: moderate, 3: severe, 4: very severe. The findings are: Heart Size, Pulmonary Congestion, Pleural Effusion Right, Pleural Effusion Left, Pulmonary Opacities Right, Pulmonary Opacities Left, Atelectasis Right, Atelectasis Left. Please use the following format for your response: " \
"Heart Size: <value>, Pulmonary Congestion: <value>, Pleural Effusion Right: <value>, Pleural Effusion Left: <value>, Pulmonary Opacities Right: <value>, Pulmonary Opacities Left: <value>, Atelectasis Right: <value>, Atelectasis Left: <value>."

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Image attribution: Stillwaterising, CC0, via Wikimedia Commons
image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_message}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image", "image": image}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=128, do_sample=True, num_beams=5)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
