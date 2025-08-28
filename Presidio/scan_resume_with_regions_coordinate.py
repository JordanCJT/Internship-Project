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

# === Define original layout size used for region design ===
BASE_WIDTH = 1240
BASE_HEIGHT = 1750

# === Define Predefined Regions ===
predefined_regions = {
    "Name": (481, 18, 1230, 152),
    "Course": (481, 161, 1230, 284),
    "Phone Number": (70, 546, 353, 573),
    "Email1": (70, 599, 353, 630),
    "Location": (70, 655, 353, 684),
    "Email2": (70, 712, 353, 749),
    "Skills": (57, 864, 364, 1196),
    "Languages": (57, 1254, 364, 1406),
    "Reference": (9, 1466, 359, 1739),
    "Profile": (444, 423, 1221, 848),
    "Work Experience": (444, 928, 1221, 1383),
    "Education": (444, 1458, 1221, 1740)
}

# === Select Multiple Files ===
root = tk.Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(
    title="Select Resume Images or PDFs",
    filetypes=[("Supported Files", "*.jpg *.png *.jpeg *.bmp *.tiff *.webp *.pdf")]
)
if not file_paths:
    print("❌ No files selected.")
    exit()

# === Loop Through All Selected Files ===
for file_path in file_paths:
    print(f"\n📂 Processing: {file_path}")
    pages = []

    if file_path.lower().endswith(".pdf"):
        print("📄 PDF detected. Converting pages to images...")
        pages = convert_from_path(file_path, dpi=300, poppler_path=POPPLER_PATH)
    else:
        pages = [Image.open(file_path)]

    for page_index, pil_image in enumerate(pages):
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # === OCR full text extraction ===
        custom_config = r'--psm 4'
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

        # === Scale predefined regions ===
        img_h, img_w = image.shape[:2]
        x_scale = img_w / BASE_WIDTH
        y_scale = img_h / BASE_HEIGHT
        scaled_regions = {
            label: (
                int(x1 * x_scale), int(y1 * y_scale),
                int(x2 * x_scale), int(y2 * y_scale)
            )
            for label, (x1, y1, x2, y2) in predefined_regions.items()
        }

        # === OCR word-level data ===
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
        region_texts = {label: [] for label in scaled_regions}

        for _, row in data.iterrows():
            if row['conf'] > 30 and isinstance(row['text'], str) and row['text'].strip():
                x, y, w, h = int(row['left']), int(row['top']), int(row['width']), int(row['height'])
                word_center = (x + w // 2, y + h // 2)
                for label, (rx1, ry1, rx2, ry2) in scaled_regions.items():
                    if rx1 <= word_center[0] <= rx2 and ry1 <= word_center[1] <= ry2:
                        region_texts[label].append(row['text'])
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # === Print grouped results ===
        print("\n[📌 Grouped Text by Region]")
        for label, words in region_texts.items():
            print(f"{label}: {' '.join(words).strip()}")

        # === Export to JSON ===
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        json_path = f"grouped_output_{base_filename}_page{page_index + 1}.json"
        grouped_output = {label: ' '.join(words).strip() for label, words in region_texts.items()}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(grouped_output, f, indent=4, ensure_ascii=False)
        print(f"✅ Grouped data saved to: {json_path}")

        # === Draw labeled region boxes ===
        for label, (x1, y1, x2, y2) in scaled_regions.items():
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # === Save and open result image ===
        output_img = f"output_{base_filename}_page{page_index + 1}.png"
        cv2.imwrite(output_img, image)
        print(f"✅ Output image saved to: {output_img}")
        os.startfile(output_img)