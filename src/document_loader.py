
from pathlib import Path
from io import BytesIO

import pymupdf
import pdfplumber
import pytesseract

from PIL import Image
from langchain_core.documents import Document


MIN_TEXT_LENGTH = 40


def load_pdf_documents(pdf_path):
    """Extract page text, tables, and OCR text from a PDF."""

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    if pdf_path.stat().st_size == 0:
        raise ValueError("The PDF file is empty.")

    documents = []

    try:
        pdf = pymupdf.open(str(pdf_path))
    except Exception as error:
        raise ValueError(
            "Could not open the PDF. It may be corrupted or encrypted."
        ) from error

    try:
        if pdf.needs_pass:
            raise ValueError("Password-protected PDFs are not supported.")

        if len(pdf) == 0:
            raise ValueError("The PDF contains no pages.")

        with pdfplumber.open(str(pdf_path)) as table_pdf:
            for page_index, page in enumerate(pdf):
                text = page.get_text("text").strip()
                extraction_method = "text"

                # Use OCR if the page has very little extractable text.
                if len(text) < MIN_TEXT_LENGTH:
                    try:
                        pixmap = page.get_pixmap(
                            matrix=pymupdf.Matrix(2, 2),
                            alpha=False
                        )
                        image = Image.open(
                            BytesIO(pixmap.tobytes("png"))
                        )

                        ocr_text = pytesseract.image_to_string(
                            image, lang="eng"
                        ).strip()

                        if len(ocr_text) > len(text):
                            text = ocr_text
                            extraction_method = "OCR"

                    except pytesseract.TesseractNotFoundError:
                        if not text:
                            raise RuntimeError(
                                "Tesseract OCR is missing. Install it "
                                "to process scanned PDFs."
                            )

                # Extract tables from the same page.
                tables_text = []
                if page_index < len(table_pdf.pages):
                    table_page = table_pdf.pages[page_index]

                    try:
                        tables = table_page.extract_tables() or []
                        for table_number, table in enumerate(tables, 1):
                            rows = []
                            for row in table:
                                cells = [
                                    str(cell).strip() if cell is not None else ""
                                    for cell in row
                                ]
                                if any(cells):
                                    rows.append(" | ".join(cells))

                            if rows:
                                tables_text.append(
                                    f"Table {table_number}:\n"
                                    + "\n".join(rows)
                                )
                    except Exception as error:
                        print(
                            f"Table extraction failed on page "
                            f"{page_index + 1}: {error}"
                        )

                combined_text = text.strip()

                if tables_text:
                    combined_text += "\n\n" + "\n\n".join(tables_text)

                if combined_text.strip():
                    documents.append(
                        Document(
                            page_content=combined_text,
                            metadata={
                                "source": pdf_path.name,
                                "page": page_index,
                                "page_number": page_index + 1,
                                "extraction_method": extraction_method,
                                "has_tables": bool(tables_text),
                            },
                        )
                    )

    finally:
        pdf.close()

    if not documents:
        raise ValueError(
            "No readable text or tables found in this PDF."
        )

    return documents
