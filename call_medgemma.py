from transformers import pipeline
from PIL import Image
import requests
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-it",
    torch_dtype=torch.bfloat16,
    device="cuda",
)

# Image attribution: Stillwaterising, CC0, via Wikimedia Commons
image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)

prompt = "You are a board-certified radiologist tasked with interpreting a chest X-ray. Follow a systematic diagnostic approach to generate a clear, structured report. Use the reasoning steps below to guide your interpretation: Verify patient and image details: Check patient ID, age, sex, study date, and imaging views (PA/AP/lateral). Mention if prior imaging is available for comparison. Assess image quality: Comment on image rotation, inspiration, penetration, and field of view. Note any limitations affecting interpretation. Review anatomy systematically using ABCDE: A (Airways & Mediastinum): Assess tracheal alignment, carina, and mediastinal contours. B (Bones & Soft Tissues): Look for fractures, deformities, or abnormal densities in ribs, clavicles, spine, or soft tissues. C (Cardiac Silhouette): Evaluate heart size, borders, and pulmonary vasculature. D (Diaphragm & Pleura): Inspect diaphragm position, costophrenic angles, and signs of pneumothorax or pleural effusion. E (Lung Fields): Identify any opacities, consolidations, nodules, interstitial markings, or asymmetry. Check for devices and foreign bodies: Describe the presence and positioning of tubes, lines, pacemakers, orthopedic implants, etc. Compare with prior studies: Describe any changes relative to previous imaging, if available. Summarize findings and impressions: Provide a structured Findings section describing observations in objective terms. Conclude with a concise Impression that synthesizes key findings and offers a clinical interpretation or recommendation. Your output should follow this format: Findings: \n [Detailed anatomic observations] \n  Impression: \n [Summary diagnosis or assessment with clinical relevance] \n Ensure clarity, conciseness, and clinical relevance in your report."

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert radiologist."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image},
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
