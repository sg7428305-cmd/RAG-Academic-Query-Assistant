from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DOCUMENT_FOLDER = "data/documents"
VECTORSTORE_FOLDER = "data/vectorstore"


def load_documents():
    documents = []

    folder = Path(DOCUMENT_FOLDER)

    for pdf_file in folder.glob("*.pdf"):
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        pdf_documents = loader.load()

        documents.extend(pdf_documents)

    return documents


def create_vector_store():

    print("\n1. Loading documents...")

    documents = load_documents()

    print(f"Total pages loaded: {len(documents)}")

    if not documents:
        print("No PDF found in data/documents")
        return

    print("\n2. Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    print("\n3. Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("\n4. Creating FAISS vector database...")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    Path(VECTORSTORE_FOLDER).mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(VECTORSTORE_FOLDER)

    print("\n✅ Vector database created successfully!")

    print(f"Saved at: {VECTORSTORE_FOLDER}")


if __name__ == "__main__":
    create_vector_store()