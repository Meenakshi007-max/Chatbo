import os
import json
from pypdf import PdfReader
from chunker import semantic_chunk_pages


INPUT_FOLDER = "data"
OUTPUT_FOLDER = "outputs"


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({
                "page": i + 1,
                "text": text
            })

    return pages


def process_all_pdfs():
    if not os.path.exists(INPUT_FOLDER):
        print("❌ data folder missing")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("❌ No PDFs found")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(INPUT_FOLDER, pdf_file)
        print(f"📄 Processing: {pdf_file}")

        pages = extract_text_from_pdf(pdf_path)

        if not pages:
            print("⚠️ No text extracted")
            continue

        chunks = semantic_chunk_pages(pages, source=pdf_file)

        output_file = pdf_file.replace(".pdf", "_chunks.json")
        output_path = os.path.join(OUTPUT_FOLDER, output_file)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    process_all_pdfs()