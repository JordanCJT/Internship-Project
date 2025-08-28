import cv2
import pytesseract
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from tkinter import filedialog
import tkinter as tk
import json
import os
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

# === Set Tesseract Path ===
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# === Set Poppler Path ===
POPPLER_PATH = r"C:\Program Files\Release-24.08.0-0\poppler-24.08.0\Library\bin"

# === Load reusable region template from Label Studio JSON ===
LABELSTUDIO_JSON = r"C:\Users\User\Downloads\PresidioResumeScanner\labelstudio_regions.json"  # ← Update this path

with open(LABELSTUDIO_JSON, "r", encoding="utf-8") as f:
    labelstudio_data = json.load(f)

# === Extract regions from the first labeled example ===
template_entry = labelstudio_data[0]  # Use the first (or one you want as template)
template_regions = []

for result in template_entry.get("annotations", [])[0].get("result", []):
    if result["type"] != "rectanglelabels":
        continue
    label = result["value"]["rectanglelabels"][0]
    x_pct, y_pct = result["value"]["x"], result["value"]["y"]
    w_pct, h_pct = result["value"]["width"], result["value"]["height"]
    img_w, img_h = result["original_width"], result["original_height"]

    x1 = int(x_pct / 100 * img_w)
    y1 = int(y_pct / 100 * img_h)
    x2 = int((x_pct + w_pct) / 100 * img_w)
    y2 = int((y_pct + h_pct) / 100 * img_h)

    template_regions.append({"label": label, "box": (x1, y1, x2, y2)})

print(f"✅ Loaded {len(template_regions)} template regions from Label Studio.")

# === Select Resume Files ===
root = tk.Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(
    title="Select Resume Images or PDFs",
    filetypes=[("Supported Files", "*.jpg *.png *.jpeg *.bmp *.tiff *.webp *.pdf")]
)
if not file_paths:
    print("❌ No files selected.")
    exit()

# === Process Each File ===
for file_path in file_paths:
    print(f"\n📂 Processing: {file_path}")
    base_filename = os.path.basename(file_path)
    pages = []

    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path, dpi=300, poppler_path=POPPLER_PATH)
    else:
        pages = [Image.open(file_path)]

    for page_index, pil_image in enumerate(pages):
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # === OCR full text extraction ===
        custom_config = r'--oem 3 --psm 4 -c preserve_interword_spaces=1'
        ocr_text = pytesseract.image_to_string(image, config=custom_config)

        print("\n[📝 OCR Extracted Text]")
        print(ocr_text)

        # === Detect PII ===
        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=ocr_text, language="en")
        print("\n[🔍 PII Entities Detected]")
        for entity in results:
            print(f"{entity.entity_type}: {ocr_text[entity.start:entity.end]} (Score: {entity.score:.2f})")
        anonymizer = AnonymizerEngine()
        anonymized = anonymizer.anonymize(text=ocr_text, analyzer_results=results)
        print("\n[🔐 Anonymized Text]")
        print(anonymized.text)

        # === Reuse same regions for all input files ===
        regions = template_regions.copy()
        region_texts = {region["label"]: [] for region in regions}

        # === OCR word-level detection ===
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)

        for _, row in ocr_data.iterrows():
            if row['conf'] > 30 and isinstance(row['text'], str) and row['text'].strip():
                x, y, w, h = int(row['left']), int(row['top']), int(row['width']), int(row['height'])
                cx, cy = x + w // 2, y + h // 2
                for region in regions:
                    x1, y1, x2, y2 = region["box"]
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        region_texts[region["label"]].append(row['text'])
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # === Print grouped text ===
        print("\n[📌 Grouped Text by Region]")
        for label, words in region_texts.items():
            print(f"{label}: {' '.join(words).strip()}")

        # === Save JSON output ===
        output_json = f"grouped_output_{base_filename}_page{page_index + 1}.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({k: ' '.join(v).strip() for k, v in region_texts.items()}, f, indent=4)
        print(f"✅ JSON saved: {output_json}")

        # === Draw region boxes ===
        for region in regions:
            x1, y1, x2, y2 = region["box"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, region["label"], (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # === Save image ===
        output_img = f"output_{base_filename}_page{page_index + 1}.png"
        cv2.imwrite(output_img, image)
        print(f"✅ Output image saved: {output_img}")
        os.startfile(output_img)
