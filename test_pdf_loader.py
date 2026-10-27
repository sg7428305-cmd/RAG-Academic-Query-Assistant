from pathlib import Path
from src.document_loader import load_pdf_documents

pdf_path = Path("data/documents/DBMS.pdf")

try:
    documents = load_pdf_documents(pdf_path)

    print("PDF processing successful!")
    print("Pages extracted:", len(documents))

    for doc in documents[:5]:
        print("\nPage:", doc.metadata["page_number"])
        print("Method:", doc.metadata["extraction_method"])
        print("Tables found:", doc.metadata["has_tables"])
        print("Text preview:", doc.page_content[:250])

except Exception as error:
    print("PDF processing failed:", error)