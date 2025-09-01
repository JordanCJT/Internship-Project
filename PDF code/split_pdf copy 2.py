import fitz  # PyMuPDF
import os

# Input and output folder setup
input_folder = r"D:\Internship\TWU"
output_base_folder = r"D:\Internship\KK Result"
os.makedirs(output_base_folder, exist_ok=True)

# Get all PDF files in the input folder
pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]

if not pdf_files:
    print("No PDF files found in the input folder.")
    exit()

# Process each PDF
for pdf_file in pdf_files:
    pdf_path = os.path.join(input_folder, pdf_file)
    print(f"Splitting file: {pdf_file}")

    pdf_name = os.path.splitext(pdf_file)[0]
    # pdf_output_folder = os.path.join(output_base_folder, pdf_name)
    # os.makedirs(pdf_output_folder, exist_ok=True)

    # Load original PDF
    doc = fitz.open(pdf_path)

    # Split pages
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        rect = page.rect
        rotation = 0

        if rect.width > rect.height:
                rotation = -90

        new_doc = fitz.open() 
        new_doc.insert_pdf(doc, from_page=page_number, to_page=page_number)

        new_doc[0].set_rotation(rotation)
        
        output_pdf_path = os.path.join(output_base_folder, f"{pdf_name}_page_{page_number + 1}.pdf")
        new_doc.save(output_pdf_path)
        new_doc.close()

    # print while still open
    print(f"{pdf_file} split into {len(doc)} pages. Saved in {output_base_folder}")
    doc.close()

print("All PDF splitting completed.")
