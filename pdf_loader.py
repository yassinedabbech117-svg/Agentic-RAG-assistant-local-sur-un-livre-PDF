
import fitz


def load_pdf(pdf_path):
    pages = []

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    if not pages:
        raise ValueError(
            "Aucun texte extrait du PDF. Un OCR peut être nécessaire."
        )

    return pages
