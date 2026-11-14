
from pypdf import PdfReader
from pathlib import Path


def load_pdf(file_path):
    """Extract text from a PDF file."""

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(str(pdf_path))

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append({
                "page": page_number,
                "text": text
            })

    return documents


if __name__ == "__main__":
    pdf_file = "documents/sample.pdf"

    pages = load_pdf(pdf_file)

    print("PDF loaded successfully!")
    print("Total pages extracted:", len(pages))

    for page in pages:
        print(f"\n--- Page {page['page']} ---")
        print(page["text"][:500])
