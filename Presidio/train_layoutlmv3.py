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

# === Define LABELS ===
LABELS = [
    'O', 'B-NAME', 'B-COURSE', 'B-PHONE_NUMBER',
    'B-EMAIL1', 'B-LOCATION', 'B-EMAIL2',
    'B-SKILLS', 'B-LANGUAGE', 'B-REFERENCE',
    'B-PROFILE', 'B-WORK_EXPERIENCE', 'B-EDUCATION',
    'B-PAST_PROJECT', 'B-LINKEDIN', 'B-AWARD',
    'B-NATIONALITY'
]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

# === Paths ===
dataset_json_path = r"C:\Users\User\Downloads\PresidioResumeScanner\layoutlmv3_dataset.json"
resume_image_folder = r"C:\Users\User\Downloads\PresidioResumeScanner\PresidioResumeScanner\resume_images"
output_dir = "./layoutlmv3-resume-model"

# === Load Dataset ===
with open(dataset_json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Filter entries with missing images
filtered_data = []
for entry in raw_data:
    image_path = os.path.join(resume_image_folder, entry.get("image_file", ""))
    if os.path.exists(image_path):
        filtered_data.append(entry)
    else:
        print(f"⚠️ Missing image skipped: {image_path}")

if not filtered_data:
    raise ValueError("❌ No valid entries with matching image files found.")

dataset = Dataset.from_list(filtered_data)

# === Load Model and Processor ===
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(LABELS),
    label2id=label2id,
    id2label=id2label
)

# === Encode function ===
def encode(example):
    image_path = os.path.join(resume_image_folder, example["image_file"])
    image = Image.open(image_path).convert("RGB")

    encoding = processor(
        images=image,
        words=example["tokens"],
        boxes=example["bboxes"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    seq_length = encoding["input_ids"].shape[1]
    labels = [label2id[label] for label in example["labels"]]
    labels += [label2id["O"]] * (seq_length - len(labels))

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "bbox": encoding["bbox"].squeeze(0),
        "pixel_values": encoding["pixel_values"].squeeze(0),
        "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0),
        "labels": torch.tensor(labels)
    }

# === Encode Dataset ===
encoded_dataset = dataset.map(encode)

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    num_train_epochs=5,
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

# === Save Model ===
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

print(f"\n✅ Training complete! Model saved to: {output_dir}")
