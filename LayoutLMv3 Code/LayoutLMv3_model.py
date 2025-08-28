import pytesseract
from PIL import Image, ImageDraw
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from collections import defaultdict
import json
import re

# === Config ===
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
image_path = r"D:\Internship\Model\Resume Data Set\b02853b9-Chin_Mei_Ling_Resume.jpg"

# === Load fine-tuned model & processor ===
model_dir = r"D:\Internship\LLM+py3.9\layoutlmv3-resume-model"
# model_dir = r"D:\Internship\LLM+py3.9\layoutlmv3-resume-model\checkpoint-200"
# D:\Internship\LLM+py3.9\layoutlmv3-resume-model\checkpoint-200
processor = LayoutLMv3Processor.from_pretrained(model_dir, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)  # will load model.safetensors
id2label = model.config.id2label

# === OCR with Tesseract ===
image = Image.open(image_path).convert("RGB")
ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

words, boxes = [], []
for i in range(len(ocr_data["text"])):
    if ocr_data["text"][i].strip():  # don't drop by confidence, just skip blanks
        words.append(ocr_data["text"][i])
        (x, y, w, h) = (ocr_data["left"][i], ocr_data["top"][i],
                        ocr_data["width"][i], ocr_data["height"][i])
        boxes.append([x, y, x + w, y + h])

print(f"✅ OCR extracted {len(words)} words")

# Normalize boxes (0–1000 scale)
w, h = image.size
boxes_normalized = [[
    int(1000 * (x1 / w)),
    int(1000 * (y1 / h)),
    int(1000 * (x2 / w)),
    int(1000 * (y2 / h))
] for (x1, y1, x2, y2) in boxes]

# === Encode for model ===
inputs = processor(
    image,
    words,
    boxes=boxes_normalized,
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

print("✅ First 20 predicted labels:", pred_labels[:20])

# === Draw predictions on image ===
draw = ImageDraw.Draw(image)
for word, box, label in zip(words, boxes, pred_labels[:len(words)]):
    if label != "O":  # draw only entities
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1] - 10), f"{word} ({label})", fill="red")

image.save("resume_with_predictions.png")
print("✅ Output saved as resume_with_predictions.png")

# === Convert predictions into structured JSON ===
entities = defaultdict(list)

for word, label in zip(words, pred_labels[:len(words)]):
    if label != "O":
        # collapse B-/I- to same field
        label = label.replace("B-", "").replace("I-", "")
        entities[label].append(word)

# Join tokens into full strings
resume_data = {k: " ".join(v) for k, v in entities.items()}

# === Regex fallback for critical fields ===
all_text = " ".join(words)

# --- Phone number (Malaysia specific, starts with 01 and 10-11 digits)
if "PHONE_NO" not in resume_data:
    phone_matches = re.findall(r'\b01[0-9][- ]?[0-9]{7,8}\b', all_text)
    if phone_matches:
        resume_data["PHONE_NO"] = phone_matches[0]

# --- Email extraction (collect full email tokens)
if "EMAIL" not in resume_data:
    email_tokens = [w for w in words if "@" in w]
    if email_tokens:
        # join around @ (handles cases like 'chin . mei @ gmail . com')
        email_candidate = "".join(email_tokens)
        email_candidate = email_candidate.replace(" ", "").replace("..", ".")
        email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.(com|my|edu\.my|org|mail|apu|)', email_candidate)
        if email_match:
            resume_data["EMAIL"] = email_match.group(0)

# --- Name heuristic: first 2-4 uppercase tokens (before 'Profile'/'Contact')
if "NAME" not in resume_data:
    candidate_tokens = []
    for w in words[:10]:  # look only at first 10 words
        if w.isupper() and len(w) > 1:
            candidate_tokens.append(w)
        else:
            break
    if candidate_tokens:
        resume_data["NAME"] = " ".join(candidate_tokens).title()

# --- Extract SKILLS
if "SKILLS" not in resume_data and "INTRODUCTION" in resume_data:
    skills_match = re.search(r'SKILLS(.*?)(WORK EXPERIENCE|LANGUAGES|$)', resume_data["INTRODUCTION"], re.DOTALL)
    if skills_match:
        skills_text = skills_match.group(1)
        skills_list = re.findall(r'[A-Za-z+()]+', skills_text)
        resume_data["SKILLS"] = [s for s in skills_list if len(s) > 2]
        # remove from INTRODUCTION
        resume_data["INTRODUCTION"] = resume_data["INTRODUCTION"].replace(skills_match.group(0), "").strip()

# --- Extract LANGUAGES
if "LANGUAGES" not in resume_data and "WORK_EXPERIENCE" in resume_data:
    lang_match = re.search(r'LANGUAGES(.*)', resume_data["WORK_EXPERIENCE"], re.DOTALL)
    if lang_match:
        lang_text = lang_match.group(1)
        langs = re.findall(r'[A-Za-z]+ \((?:Fluent|Basic|Intermediate)\)', lang_text)
        resume_data["LANGUAGES"] = langs
        # remove from WORK_EXPERIENCE
        resume_data["WORK_EXPERIENCE"] = resume_data["WORK_EXPERIENCE"].replace(lang_match.group(0), "").strip()


# --- Clean up noise: remove phone/email from education/introduction
for field in ["EDUCATION_HISTORY", "INTRODUCTION"]:
    if field in resume_data:
        resume_data[field] = re.sub(r'\b01[0-9][- ]?[0-9]{7,8}\b', '', resume_data[field])
        resume_data[field] = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.(com|my|edu\.my|org)', '', resume_data[field])
        resume_data[field] = resume_data[field].strip()

# Save JSON
with open("resume_extracted.json", "w", encoding="utf-8") as f:
    json.dump(resume_data, f, indent=2, ensure_ascii=False)

print("✅ Extracted JSON saved as resume_extracted.json")

