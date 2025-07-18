import os
import pandas as pd
from transformers import pipeline
from PIL import Image
import torch
import json


# Load metadata
metadata_path = "data/keno_1000/metadata.csv"
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

# Initialize the pipeline
pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-it",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

example_output = "Reasoning:\nStep: Look for therapy devices.\nAction: A tubular radiopaque foreign body with radiopaque markings is observed on the right side. It enters the chest in the middle of the image and then runs to the right apical region. It measures roughly one centimeter in thickness. The tip of the structure shows small holes. Altogether, a tubular structure entering the right chest cavity and ending in the apical region is typical for a pleural drainage. Adjacent to where the pleural tube enters the chest, small linear, hyperdense structures are seen. The appear to be cutaneous staple suture lines in the right thoracic region. These findings indicate a postoperative state with pleural drainage tubes in place. I am now looking at the left and right jugular region, but I do not see any central catheter placed. Looking at the neck, I also do not see any tracheal tubes. I am now looking at the hilar region, but I am seeing no therapy devices or clips from previous surgery. Overall, I cannot find any more therapy devices, besides the pleural drainage.\nStep: Assess the cardiomediastinal silhouette.\nAction: The cardiomediastinum is midline. It seems the left and right diameter of the thorax, adjacent to the cardiomedastinum are symmetric. Therefore, there seems to be no shift of the cardiomediastinum. I am also looking at the clavicles to evaluate if the patient is positioned correctly. The clavicles are in symmetric distance to the sternal bone, I can therefore assume the patient is symmetric and there is no midline shift. The mediastinum does not appear enlarged. I look closer at the hilar region, to check for lumps or enlargement, However, they appear to be normal in size. I now look at the heart. The diameter of the heart seems normal, so it does not seem enlarged. I also have to consider the heart shape, but it also looks normal so overall, the heart looks normal.\nStep: Assess the lungs and pleura.\nAction: I now look at the lung parenchyma to evaluate for pneumothorax. The lung margins are in contact with the thoracic wall on all sides, and I do not see any pleural dehiscence. This suggests that no pneumothorax is present.\nStep: Confirm placement of pleural drainage tubes.\nAction: I trace the course of the pleural drainage tubes. One tip is seen in the right apical region and another in the right basal region. Both appear to be in expected positions, consistent with proper postoperative placement.\nStep: Compare ventilation and opacities between lungs.\nAction: I now compare both lungs. The right lung shows diffuse pleural-based opacities, more prominent posteriorly, while the anterior regions are less dense. I look at the left lung, which appears well ventilated without any abnormal opacities. I do not see air bronchograms, which argues against the presence of a pneumonic infiltrate.\nStep: Assess for signs of pulmonary congestion or infection.\nAction: I evaluate the hilar regions and pulmonary vasculature. I do not observe increased vascular markings or hilar enlargement. There are no thickened interlobular septa or signs of peribronchial cuffing. These findings suggest there is no pulmonary edema or bronchial inflammation.\nFinal answer:\nThe findings are consistent with a postoperative right-sided pleural effusion and associated atelectatic changes. There is no evidence of pneumonia, pulmonary congestion, or other acute thoracic pathology."
# Define the clinical reasoning prompt
prompt_base = (
    "You are a board-certified radiologist tasked with interpreting a chest X-ray image. "
    "Please follow a systematic diagnostic approach to produce a clear, structured report. "
    "Your output must follow this format:\n\n"
    "Reasoning:\n"
    "Step: [Description of where you are looking]\n"
    "Action: [Description of what you see and interpret]\n"
    "...\n"
    "Final answer:\n"
    "[Summary diagnosis or clinical impression]\n\n"
    "Ensure that your reasoning is thorough, clinically relevant, and easy to follow. "
    "Use step-by-step anatomical logic, describing where you are looking and what you are seeing. "
    "You are provided with an example below to guide the style and level of detail expected:\n\n"
    + example_output +
    "\nNow, write the report for the provided chest X-ray image and corresponding clinical data.\n\n"
)

# Annotate images
image_dir = "data/keno_1000/data_png"
output_path = "data/keno_1000/draft_annotations_with_clinical_info.json"
annotations = {}

for filename in os.listdir(image_dir):
    if not filename.lower().endswith(".png"):
        continue

    uid = filename.split(".")[0]
    image_path = os.path.join(image_dir, filename)

    if uid not in metadata_df.index:
        print(f"Skipping {filename}: UID not found in metadata.")
        continue

    try:
        row = metadata_df.loc[uid]
        clinical_info = describe_row(row)

        prompt =  prompt_base + "\n\n" + clinical_info
        image = Image.open(image_path).convert("RGB")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]}
        ]

        result = pipe(text=messages, max_new_tokens=512)
        output_text = result[0]["generated_text"][-1]["content"]
        annotations[filename] = output_text

    except Exception as e:
        print(f"Error processing {filename}: {e}")

with open(output_path, "w") as f:
    json.dump(annotations, f, indent=2)

print(f"Saved {len(annotations)} annotations to {output_path}")
