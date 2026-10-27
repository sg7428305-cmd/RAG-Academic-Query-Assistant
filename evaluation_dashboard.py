
import json
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
REPORT_FILE = BASE_DIR / "evaluation_report.json"

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("RAG Academic Query Assistant")
st.subheader("Retrieval Evaluation Dashboard")

st.write(
    "Review retrieval performance using the latest saved "
    "evaluation report."
)

if not REPORT_FILE.exists():
    st.warning(
        "Evaluation report nahi mili. Pehle run karo: "
        "python evaluate_retrieval.py"
    )
    st.stop()

try:
    with open(REPORT_FILE, "r", encoding="utf-8") as file:
        report = json.load(file)
except (json.JSONDecodeError, OSError) as error:
    st.error(f"Report read nahi ho paayi: {error}")
    st.stop()


def find_value(data, possible_keys, default=None):
    """Find a value using one of the supported key names."""
    if not isinstance(data, dict):
        return default

    for key in possible_keys:
        if key in data:
            return data[key]

    return default


# Support common report field names.
tests = find_value(
    report,
    ["results", "tests", "test_results", "evaluations"],
    []
)

if not isinstance(tests, list):
    tests = []

total_tests = find_value(
    report,
    ["tests_run", "total_tests", "total_questions"],
    len(tests)
)

hit_count = find_value(
    report,
    ["expected_file_hits", "hits", "successful_retrievals"],
    None
)

hit_rate = find_value(
    report,
    ["retrieval_hit_rate_percent", "hit_rate_percent", "hit_rate"],
    None
)

indexed_pdfs = find_value(
    report,
    ["indexed_pdfs", "total_pdfs", "documents_count"],
    None
)

# Derive metrics from individual results when possible.
if hit_count is None and tests:
    hit_count = sum(
        1 for item in tests
        if isinstance(item, dict)
        and item.get("retrieval_hit") is True
    )

if hit_count is None:
    hit_count = 0

if hit_rate is None and total_tests:
    hit_rate = round(100 * hit_count / total_tests, 2)

st.caption(f"Report file: {REPORT_FILE.name}")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Test Questions", total_tests)
col2.metric("Expected-File Hits", hit_count)
col3.metric(
    "Retrieval Hit Rate",
    f"{hit_rate}%" if hit_rate is not None else "N/A"
)
col4.metric(
    "Indexed PDFs",
    indexed_pdfs if indexed_pdfs is not None else "N/A"
)

st.divider()
st.subheader("Test Results")

if tests:
    rows = []

    for item in tests:
        if not isinstance(item, dict):
            continue

        question = find_value(
            item, ["question", "query"], "Unknown question"
        )

        expected_file = find_value(
            item, ["expected_file", "expected_source"], "N/A"
        )

        retrieval_hit = item.get("retrieval_hit")

        sources = find_value(
            item,
            ["retrieved_sources", "sources", "retrieved_documents"],
            []
        )

        if isinstance(sources, list):
            if sources and isinstance(sources[0], dict):
                source_names = ", ".join(
                    str(
                        source.get("filename")
                        or source.get("source")
                        or source.get("file_name")
                        or "Unknown source"
                    )
                    for source in sources
                )
            else:
                source_names = ", ".join(map(str, sources))
        else:
            source_names = str(sources)

        rows.append({
            "Question": question,
            "Expected File": expected_file,
            "Retrieval Hit": (
                "Yes" if retrieval_hit is True
                else "No" if retrieval_hit is False
                else "Not applicable"
            ),
            "Retrieved Sources": source_names
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Results as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="retrieval_evaluation.csv",
            mime="text/csv"
        )
    else:
        st.info(
            "Report mein test results ka format pehchana nahi gaya."
        )
else:
    st.info(
        "Report mein individual test results nahi mile. "
        "Neeche raw report dekh sakte ho."
    )

with st.expander("View Raw Evaluation Report"):
    st.json(report)

st.caption(
    "Note: Retrieval hit rate document retrieval ko measure karta hai, "
    "generated answers ki factual correctness ko nahi."
)
