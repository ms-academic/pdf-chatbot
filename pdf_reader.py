# pdf_text_extraction.py

import PyPDF2
import os

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

if __name__ == "__main__":
    pdf_folder = "pdfs/"
    output_folder = "extracted_text/"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(pdf_path)

            output_path = os.path.join(output_folder, filename.replace(".pdf", ".txt"))
            with open(output_path, "w") as f:
                f.write(text)
            print(f"Extracted text saved to {output_path}")
