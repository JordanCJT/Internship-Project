import os
import json
import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    default_data_collator
)

# === Define LABELS (your schema) ===
LABELS = [
    'O',
    'B-NAME',
    'B-POSITION',
    'B-WORK_EXPERIENCE',
    'B-HOME_ADDRESS',
    'B-EDUCATION_HISTORY',
    'B-PHONE_NO',
    'B-EMAIL',
    'B-TECHNICAL_SKILLS',
    'B-LANGUAGE',
    'B-REFERENCE',
    'B-INTRODUCTION',
    'B-PROJECT'
]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

# === Map dataset labels to your schema ===
LABEL_MAPPING = {
    "B-COURSE": "B-EDUCATION_HISTORY",
    "B-EMAIL1": "B-EMAIL",
    "B-EMAIL2": "B-EMAIL",
    "B-LOCATION": "B-HOME_ADDRESS",
    "B-SKILLS": "B-TECHNICAL_SKILLS",
    "B-LANGUAGES": "B-LANGUAGE",
    "B-PROFILE": "B-INTRODUCTION",
    "B-EDUCATION": "B-EDUCATION_HISTORY"
}

# === Paths ===
dataset_json_path = r"D:\Internship\LLM+py3.9\layoutlmv3_dataset.json"
resume_image_folder = r"D:\Internship\LLM+py3.9\image"
output_dir = r"D:\Internship\LLM+py3.9\layoutlmv3-resume-model"

# === Load Dataset ===
with open(dataset_json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Normalize labels in raw_data
for entry in raw_data:
    entry["labels"] = [
        LABEL_MAPPING.get(lbl, lbl if lbl in LABELS else "O")
        for lbl in entry["labels"]
    ]

# Filter entries with missing images
filtered_data = []
for entry in raw_data:
    image_path = os.path.join(resume_image_folder, entry.get("image_file", ""))
    if os.path.exists(image_path):
        filtered_data.append(entry)
    else:
        print(f"⚠ Missing image skipped: {image_path}")

if not filtered_data:
    raise ValueError("❌ No valid entries with matching image files found.")

dataset = Dataset.from_list(filtered_data)

# === Load Base Model and Processor ===
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)

# === Encode function ===
def encode(example):
    image_path = os.path.join(resume_image_folder, example["image_file"])
    image = Image.open(image_path).convert("RGB")

    # Clip all bbox values to [0, 1000]
    clipped_bboxes = [[
        min(1000, max(0, int(coord))) for coord in bbox
    ] for bbox in example["bboxes"]]

    encoding = processor(
        image,
        example["tokens"],
        boxes=clipped_bboxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    seq_length = encoding["input_ids"].shape[1]

    # map labels to IDs
    labels = [label2id[lbl] for lbl in example["labels"]]
    labels += [label2id["O"]] * (seq_length - len(labels))

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "bbox": encoding["bbox"].squeeze(0),
        "pixel_values": encoding["pixel_values"].squeeze(0),
        "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

# === Encode Dataset ===
encoded_dataset = dataset.map(encode, remove_columns=dataset.column_names)

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    num_train_epochs=20,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    remove_unused_columns=False,
    push_to_hub=False,
    logging_steps=10
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    data_collator=default_data_collator,
)

# === Train ===
trainer.train()

# === Force Save Model ===
trainer.save_model(output_dir)
processor.save_pretrained(output_dir)

print(f"\n✅ Training complete! Model saved to: {output_dir}")
