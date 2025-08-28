import pdfplumber
import os
import json
from tkinter import Tk, filedialog

def extract_text_preserve_structure(pdf_path, output_txt_path=None):
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            lines = page.extract_text(x_tolerance=1, y_tolerance=3, layout=True)
            if lines:
                cleaned_lines = [line for line in lines.splitlines() if line.strip()]
                full_text.append("\n".join(cleaned_lines))
            else:
                full_text.append("[No text found]")

    final_text = "\n\n".join(full_text)

    if output_txt_path:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"Saved structured output to: {output_txt_path}")
    else:
        print(final_text)

    return final_text

def create_json_file(pdf_path, text):
    json_data = {
        "data": {
            "text": text
        }
    }
    json_filename = os.path.splitext(pdf_path)[0] + "_plumber_clean.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    print(f"Saved JSON output to: {json_filename}")
    return json_filename

def select_pdf_files():
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select PDF files",
        filetypes=[("PDF files", "*.pdf")]
    )
    root.destroy()
    return file_paths

# Example usage
if __name__ == "__main__":
    pdf_paths = select_pdf_files()
    for pdf_path in pdf_paths:
        output_txt_path = os.path.splitext(pdf_path)[0] + "_plumber_clean.txt"
        extracted_text = extract_text_preserve_structure(pdf_path, output_txt_path)
        create_json_file(pdf_path, extracted_text)
    print(f"Processed {len(pdf_paths)} PDF file(s).")