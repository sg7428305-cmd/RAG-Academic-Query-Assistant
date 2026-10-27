
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


VECTORSTORE_FOLDER = "data/vectorstore"


def search_documents():

    if not Path(VECTORSTORE_FOLDER, "index.faiss").exists():
        print("Vector database not found!")
        print("First run: python src/vector_store.py")
        return

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Loading vector database...")

    vector_store = FAISS.load_local(
        VECTORSTORE_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("\nAcademic Query Assistant is ready!")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Enter your question: ").strip()

        if question.lower() == "exit":
            print("Thank you for using Academic Query Assistant!")
            break

        if not question:
            print("Please enter a question.")
            continue

        results = vector_store.similarity_search_with_score(
            question,
            k=3
        )

        print("\n--- Retrieved Information ---")

        if not results:
            print("No relevant information found.")
        else:
            for index, (document, score) in enumerate(results, start=1):
                source = Path(
                    document.metadata.get("source", "Unknown")
                ).name

                page = document.metadata.get("page")
                page_label = page + 1 if isinstance(page, int) else "Unknown"

                print(f"\nResult {index}")
                print(f"Source: {source}")
                print(f"Page: {page_label}")
                print(f"Distance score: {score:.4f}")
                print("Text:")
                print(document.page_content)

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    search_documents()
