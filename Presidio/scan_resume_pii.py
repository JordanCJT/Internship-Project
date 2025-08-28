import cv2
import pytesseract
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import tkinter as tk
from tkinter import filedialog

# === Step 0: Set Tesseract path if it's not in system PATH ===
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# === Step 1: Select resume image file ===
def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Resume Image File",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")]
    )
    return file_path

image_path = select_file()
if not image_path:
    print("❌ No file selected. Exiting.")
    exit()

# === Step 2: Read image and extract text ===
image = cv2.imread(image_path)
custom_config = r'--psm 6'  # Assume a block of text layout
ocr_text = pytesseract.image_to_string(image, config=custom_config)

print("\n[📝 OCR Extracted Text]")
print(ocr_text)

# === Step 3: Detect PII entities using Presidio ===
analyzer = AnalyzerEngine()
results = analyzer.analyze(text=ocr_text, language="en")

print("\n[🔍 PII Entities Detected]")
for entity in results:
    print(f"{entity.entity_type}: {ocr_text[entity.start:entity.end]} (Score: {entity.score:.2f})")

# === Step 4: Anonymize the detected PII ===
anonymizer = AnonymizerEngine()
anonymized = anonymizer.anonymize(text=ocr_text, analyzer_results=results)

print("\n[🔐 Anonymized Text]")
print(anonymized.text)

# === Step 5: Draw green boxes around detected text regions ===
data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
confident_data = data[data['conf'] > 60]

for _, row in confident_data.iterrows():
    (x, y, w, h) = (int(row['left']), int(row['top']), int(row['width']), int(row['height']))
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# === Step 6: Save the image with drawn rectangles ===
output_image_path = "output_detected_text_regions.png"
cv2.imwrite(output_image_path, image)
print(f"\n[✅ Saved] Output image with text boxes saved as: {output_image_path}")

# === Step 7: Show the output image in a popup window ===
cv2.imshow("Detected Text Regions", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
