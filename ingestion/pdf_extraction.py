
from pathlib import Path

import pymupdf


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        dict: A dictionary containing the PDF name and extracted text.
    """
    pdf_path = Path(pdf_path)

    doc = pymupdf.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()

    return {
        "name": pdf_path.name,
        "text": text
    }

