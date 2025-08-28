import sys
import os
import cv2
import torch
import numpy as np
import json
from collections import OrderedDict
import pytesseract
from pdf2image import convert_from_path
import tkinter as tk
from tkinter import filedialog, messagebox

# === Load CRAFT modules ===
sys.path.append(os.path.join(os.path.dirname(__file__), 'CRAFT-pytorch'))
from craft import CRAFT
import imgproc
from craft_utils import getDetBoxes, adjustResultCoordinates

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Image or PDF File",
        filetypes=[("Image or PDF Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp *.pdf")]
    )
    return file_path

def select_files():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select Image or PDF Files",
        filetypes=[("Image or PDF Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp *.pdf")]
    )
    return list(file_paths)

def select_files_or_folder():
    import tkinter as tk
    from tkinter import filedialog, messagebox
    root = tk.Tk()
    root.withdraw()
    # Show a simple dialog to choose between files or folder
    response = messagebox.askquestion(
        "Input Selection",
        "Do you want to select a folder? (Click 'No' to select files)",
        icon='question'
    )
    if response == 'yes':
        folder_path = filedialog.askdirectory(title="Select Folder Containing Images or PDFs")
        if not folder_path:
            print("No folder selected. Exiting.")
            exit()
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.pdf')
        file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(exts)]
        return file_paths
    else:
        file_paths = filedialog.askopenfilenames(
            title="Select Image or PDF Files",
            filetypes=[("Image or PDF Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp *.pdf")]
        )
        return list(file_paths)

def clean_text(text):
    import re
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_template_regions(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    for item in data:
        if not isinstance(item, dict):
            continue
        regions = []
        for ann in item.get("annotations", []):
            for region in ann.get("result", []):
                val = region["value"]
                # Get label from value["rectanglelabels"]
                if "rectanglelabels" in val and val["rectanglelabels"]:
                    label = val["rectanglelabels"][0]
                elif "from_name" in region:
                    label = region["from_name"]
                else:
                    label = "unknown"
                orig_w = region["original_width"]
                orig_h = region["original_height"]
                if not all(k in val for k in ("x", "y", "width", "height")):
                    continue
                x1 = int(val["x"] / 100 * orig_w)
                y1 = int(val["y"] / 100 * orig_h)
                x2 = int((val["x"] + val["width"]) / 100 * orig_w)
                y2 = int((val["y"] + val["height"]) / 100 * orig_h)
                # Include orig_w and orig_h for scaling later
                regions.append((label, (x1, y1, x2, y2, orig_w, orig_h)))
        if regions:
            return regions
    return []

# === Load model ===
print("[INFO] Loading CRAFT model...")
model_path = 'CRAFT-pytorch/weights/craft_mlt_25k.pth'
net = CRAFT()
state_dict = torch.load(model_path, map_location='cpu')
net.load_state_dict(OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items()))
net.eval()

# === File selection ===
image_paths = select_files_or_folder()
if not image_paths:
    print("No files selected. Exiting.")
    exit()

# === Automatically find Label Studio JSON in the same folder ===
script_dir = os.path.dirname(os.path.abspath(__file__))
json_files = [f for f in os.listdir(script_dir) if f.lower().endswith('.json')]
if not json_files:
    print("No JSON file found in the script directory. Exiting.")
    exit()

# Automatically select region template JSON based on input file name or other logic

# === Define required region labels for each template ===
# Map template file name (without .json) to required region labels
TEMPLATE_REQUIRED_LABELS = {
    "format1": {'Profile', 'Location', 'Course', 'Email2', 'Languages', 'Work Experience', 'Education', 'Reference', 'Email1', 'Phone Number', 'Name', 'Skills'},
    "format2": {'Home Address', 'Phone No', 'Linkedln', 'Education History', 'Language', 'Technical Skills', 'Work Experience', 'Email', 'Project', 'Name'},
    "format3": {'Name', 'Technical Skills', 'Language', 'Education History', 'Position', 'Reference', 'Home Address', 'Email', 'Work Experience', 'Profile', 'Phone No'},
    # Add more as needed
}

def get_region_template_for_file(image_path, json_files):
    base = os.path.basename(image_path).lower()
    for jf in json_files:
        if os.path.splitext(jf)[0].lower() in base:
            return os.path.join(script_dir, jf)
    # No matching template found
    return None

for image_path in image_paths:
    print(f"\n[INFO] Processing file: {image_path}")
    images = []
    image_filenames = []
    if image_path.lower().endswith('.pdf'):
        print("[INFO] Converting PDF to images...")
        images = convert_from_path(image_path, dpi=600, poppler_path=r"D:\Internship\poppler-24.08.0\Library\bin")
        images = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in images]
        base = os.path.splitext(os.path.basename(image_path))[0]
        image_filenames = [f"{base}_page-{str(i+1).zfill(4)}.jpg" for i in range(len(images))]
    else:
        image = imgproc.loadImage(image_path)
        images = [image]
        image_filenames = [os.path.basename(image_path)]


    # Auto-select and load template regions for this file
    labelstudio_json = get_region_template_for_file(image_path, json_files)
    if labelstudio_json is None:
        print(f"[WARNING] No matching JSON template found for file: {image_path}")
        print("  Please add a template JSON for this resume format.")
        print("  Skipping this file.\n")
        continue
    print(f"[INFO] Using Label Studio JSON: {labelstudio_json}")
    template_regions = get_template_regions(labelstudio_json)
    # === Format validation for multiple templates ===
    template_key = os.path.splitext(os.path.basename(labelstudio_json))[0].lower()
    required_labels = TEMPLATE_REQUIRED_LABELS.get(template_key)
    found_labels = {label for label, _ in template_regions}
    if not template_regions:
        print(f"[ERROR] No regions found in the template JSON for file: {image_path}")
        print("  Skipping this file.\n")
        continue
    if required_labels is None:
        print(f"[WARNING] Template '{template_key}' is not recognized in TEMPLATE_REQUIRED_LABELS.")
        print(f"  Please add required fields for this template in the code.")
        print(f"  Skipping file: {image_path}\n")
        continue
    if not required_labels.issubset(found_labels):
        print(f"[ERROR] Format mismatch or missing required fields in file: {image_path}")
        print(f"  Template: {template_key}")
        print(f"  Required: {required_labels}")
        print(f"  Found: {found_labels}")
        print("  Skipping this file.\n")
        continue

    # Auto-create output directory for this file
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(os.path.dirname(image_path), f"{base_name}_output")
    os.makedirs(output_dir, exist_ok=True)

    all_ocr_results = {}

    # === Process each page ===
    for page_num, (image, img_filename) in enumerate(zip(images, image_filenames), 1):
        print(f"\n[INFO] Processing page {page_num} ({img_filename})...")

        img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
        ratio_h = ratio_w = 1 / target_ratio
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            y, _ = net(x)
        score_text = y[0, :, :, 0].cpu().data.numpy()
        link_refine = y[0, :, :, 1].cpu().data.numpy()
        boxes, _ = getDetBoxes(score_text, link_refine, text_threshold=0.6, link_threshold=0.3, low_text=0.3, poly=True)
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

        ocr_results = {}

        img_h, img_w = image.shape[:2]

        for name, (x1, y1, x2, y2, orig_w, orig_h) in template_regions:
            # Scale coordinates to current image size
            scale_x = img_w / orig_w
            scale_y = img_h / orig_h
            sx1 = int(x1 * scale_x)
            sy1 = int(y1 * scale_y)
            sx2 = int(x2 * scale_x)
            sy2 = int(y2 * scale_y)
            roi = image[sy1:sy2, sx1:sx2]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_sharp = cv2.GaussianBlur(roi_gray, (0, 0), 3)
            roi_sharp = cv2.addWeighted(roi_gray, 1.5, roi_sharp, -0.5, 0)

            config = '--psm 6 --oem 3'
            text = pytesseract.image_to_string(roi_sharp, config=config).strip()
            text = clean_text(text)
            print(f"[{name}]\n{text}\n")

            ocr_results[name] = text

            # Draw visual box
            cv2.rectangle(image, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)
            cv2.putText(image, name, (sx1 + 5, sy1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Store results for this page
        all_ocr_results[f"page_{page_num}"] = ocr_results

        # Auto-create filename for annotated image in output directory
        output_img_name = f'{base_name}_filtered_page_{page_num}.png'
        output_path = os.path.join(output_dir, output_img_name)
        cv2.imwrite(output_path, image)
        print(f"[INFO] Output saved: {output_path}")

    # === Warn if output JSON data is mostly empty ===
    total_fields = sum(len(page) for page in all_ocr_results.values())
    empty_fields = sum(
        sum((not v or not str(v).strip() or v.lower() == 'nan') for v in page.values())
        for page in all_ocr_results.values()
    )
    if total_fields > 0 and empty_fields / total_fields > 0.5:
        print(f"[WARNING] More than 50% of the output fields are empty or missing for file: {image_path}")
        print(f"  Please check the template, image quality, or OCR settings.\n")

    # Save all OCR results for this file to a single JSON file (overwrite if exists)
    # Auto-create filename for JSON in output directory
    json_filename = os.path.join(output_dir, f"ocr_results_{base_name}.json")
    with open(json_filename, "w", encoding="utf-8") as jf:
        json.dump(all_ocr_results, jf, ensure_ascii=False, indent=4)
    print(f"[INFO] All OCR results saved: {json_filename}")
