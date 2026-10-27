
import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "data" / "vectorstore" / "uploaded_index"
INFO_FILE = BASE_DIR / "data" / "vectorstore" / "uploaded_documents.json"
OUTPUT_FILE = BASE_DIR / "evaluation_report.json"

TESTS = [
    {
        "question": "What is the difference between AI, ML, and Deep Learning?",
        "expected_terms": ["artificial intelligence", "machine learning", "deep learning"],
        "expected_file": "AI_Machine_Learning_Test_Document.pdf",
    },
    {
        "question": "Explain precision and recall.",
        "expected_terms": ["precision", "recall"],
        "expected_file": "AI_Machine_Learning_Test_Document.pdf",
    },
    {
        "question": "What is DBMS?",
        "expected_terms": ["dbms"],
       "expected_file": "DBMS.pdf.pdf",
    },
    {
        "question": "Explain quantum teleportation using my uploaded notes.",
        "expected_terms": [],
        "expected_file": None,
    },
]


def main():
    if not (INDEX_DIR / "index.faiss").exists():
        print("Uploaded FAISS index not found.")
        print("Please upload and process your PDFs in the app first.")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    store = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    if INFO_FILE.exists():
        document_info = json.loads(
            INFO_FILE.read_text(encoding="utf-8")
        )
    else:
        document_info = {}

    reports = []

    for test in TESTS:
        results = store.similarity_search_with_score(
            test["question"],
            k=5,
        )

        sources = []
        combined_text = []

        for document, score in results:
            metadata = document.metadata or {}
            source = Path(
                str(metadata.get("file_name")
                    or metadata.get("source")
                    or "Unknown")
            ).name

            page = metadata.get("page_number")
            if page is None and metadata.get("page") is not None:
                page = metadata["page"] + 1

            sources.append({
                "file": source,
                "page": page if page is not None else "Unknown",
                "distance": round(float(score), 4),
            })
            combined_text.append(document.page_content.lower())

        retrieved_files = {item["file"] for item in sources}
        context_text = " ".join(combined_text)

        expected_file = test["expected_file"]
        hit = (
            expected_file in retrieved_files
            if expected_file
            else None
        )

        found_terms = [
            term for term in test["expected_terms"]
            if term in context_text
        ]

        reports.append({
            "question": test["question"],
            "expected_file": expected_file,
            "retrieval_hit": hit,
            "expected_terms_found": found_terms,
            "expected_terms_total": len(test["expected_terms"]),
            "source_count": len(sources),
            "sources": sources,
            "note": (
                "Retrieval hit is not proof that the generated answer is correct. "
                "The unsupported question requires a separate answer-level check."
            ),
        })

    total_known = sum(
        item["retrieval_hit"] is not None for item in reports
    )
    hits = sum(item["retrieval_hit"] is True for item in reports)

    summary = {
        "tests_run": len(reports),
        "retrieval_tests_with_expected_file": total_known,
        "retrieval_hits": hits,
        "retrieval_hit_rate_percent": (
            round(hits / total_known * 100, 2)
            if total_known else None
        ),
        "indexed_pdf_count": len(document_info),
        "results": reports,
    }

    OUTPUT_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== RETRIEVAL EVALUATION =====")
    print(f"Tests run: {len(reports)}")
    print(f"Indexed PDFs: {len(document_info)}")

    if total_known:
        print(f"Expected-file hits: {hits}/{total_known}")
        print(f"Retrieval hit rate: {summary['retrieval_hit_rate_percent']}%")

    for item in reports:
        print(f"\nQuestion: {item['question']}")
        print(f"Expected file: {item['expected_file'] or 'Not applicable'}")
        print(f"Retrieval hit: {item['retrieval_hit']}")
        print(f"Retrieved sources: {item['source_count']}")

        for source in item["sources"]:
            print(
                f"  - {source['file']}, page {source['page']}, "
                f"distance {source['distance']}"
            )

    print(f"\nFull report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
