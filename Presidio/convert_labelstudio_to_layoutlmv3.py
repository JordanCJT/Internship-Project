import json
import pytesseract
import cv2
import os
from tqdm import tqdm

# === Set up Tesseract path ===
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# === Customize paths ===
labelstudio_json_path = r"C:\Users\User\Downloads\PresidioResumeScanner\project-10-at-2025-08-22-11-57-ff7c134a.json"
image_folder = r"C:\Users\User\Downloads\PresidioResumeScanner\PresidioResumeScanner\resume_images"
output_path = "layoutlmv3_dataset.json"

# === Label Studio to LayoutLMv3 label mapping ===
def convert_label(label):
    return f"B-{label.upper().replace(' ', '_')}"

# === Load exported Label Studio JSON ===
if not os.path.exists(labelstudio_json_path):
    raise FileNotFoundError(f"❌ Cannot find label file at: {labelstudio_json_path}")

with open(labelstudio_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = []

for task in tqdm(data, desc="🔄 Processing labeled resumes"):
    # Resolve full image path
    image_name = os.path.basename(task["image"])
    image_path = os.path.join(image_folder, image_name)

    if not os.path.exists(image_path):
        print(f"⚠️ Image not found: {image_path} — skipping.")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ OpenCV could not read image: {image_path} — skipping.")
        continue

    # === Run OCR with Tesseract ===
    try:
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f"❌ OCR failed on image: {image_path} — {str(e)}")
        continue

    tokens, bboxes, labels = [], [], []
    img_h, img_w = image.shape[:2]

    for i in range(len(ocr_data["text"])):
        word = ocr_data["text"][i].strip()
        if not word or int(ocr_data["conf"][i]) < 60:
            continue

        x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
        box_center = (x + w / 2, y + h / 2)

        assigned_label = "O"  # Default label

        for region in task.get("label", []):
            label = region["rectanglelabels"][0]

            # Convert percentage → absolute pixel coords
            x1 = int(region["x"] / 100 * region["original_width"])
            y1 = int(region["y"] / 100 * region["original_height"])
            x2 = int((region["x"] + region["width"]) / 100 * region["original_width"])
            y2 = int((region["y"] + region["height"]) / 100 * region["original_height"])

            # Scale to actual OpenCV image size
            scale_x = img_w / region["original_width"]
            scale_y = img_h / region["original_height"]
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

            if x1 <= box_center[0] <= x2 and y1 <= box_center[1] <= y2:
                assigned_label = convert_label(label)
                break

        tokens.append(word)
        bboxes.append([x, y, x + w, y + h])
        labels.append(assigned_label)

    if tokens:
        samples.append({
            "tokens": tokens,
            "bboxes": bboxes,
            "labels": labels,
            "image_file": image_name
        })
    else:
        print(f"⚠️ No valid tokens extracted for: {image_path}")

# === Save as LayoutLMv3 training JSON ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=4, ensure_ascii=False)

print(f"\n✅ Dataset saved to: {output_path} ({len(samples)} samples)")
