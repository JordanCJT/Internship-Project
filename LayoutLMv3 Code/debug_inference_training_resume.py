import os
import json
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# === Paths ===
dataset_json_path = r"D:\Internship\LLM+py3.9\layoutlmv3_dataset.json"
resume_image_folder = r"D:\Internship\LLM+py3.9\image"
model_dir = r"D:\Internship\LLM+py3.9\layoutlmv3-resume-model"

# === Load dataset JSON ===
with open(dataset_json_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Just take the first resume for debugging
sample = dataset[1]  # Change index as needed
image_path = os.path.join(resume_image_folder, sample["image_file"])
print(f"✅ Debugging on training resume: {image_path}")

# === Load model and processor ===
processor = LayoutLMv3Processor.from_pretrained(model_dir, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
id2label = model.config.id2label

# === Prepare inputs ===
image = Image.open(image_path).convert("RGB")
clipped_bboxes = [[
    min(1000, max(0, int(coord))) for coord in bbox
] for bbox in sample["bboxes"]]

inputs = processor(
    image,
    sample["tokens"],
    boxes=clipped_bboxes,
    return_tensors="pt",
    padding="max_length",
    truncation=True
)

# === Predict ===
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(-1)

pred_ids = predictions[0].tolist()
pred_labels = [id2label[i] for i in pred_ids]

# === Collect entities into JSON ===
from collections import defaultdict
entities = defaultdict(list)
for word, label in zip(sample["tokens"], pred_labels[:len(sample["tokens"])]):
    if label != "O":
        label = label.replace("B-", "")  # normalize label
        entities[label].append(word)

resume_data = {k: " ".join(v) for k, v in entities.items()}

print("\n✅ Extracted from training resume:")
print(json.dumps(resume_data, indent=2, ensure_ascii=False))
