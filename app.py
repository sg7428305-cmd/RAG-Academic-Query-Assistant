
import json
import hashlib
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ask_question import answer_question
from src.document_loader import load_pdf_documents


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "data" / "documents" / "uploads"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore" / "uploaded_index"
DOCUMENTS_INFO = BASE_DIR / "data" / "vectorstore" / "uploaded_documents.json"
CHAT_HISTORY_FILE = BASE_DIR / "data" / "vectorstore" / "chat_sessions.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.parent.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Enterprise Document Intelligence",
    page_icon="📚",
    layout="wide",
)


# --------------------------------------------------
# 2. EMBEDDINGS
# --------------------------------------------------

@st.cache_resource
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# 3. DOCUMENT STORAGE
# --------------------------------------------------

def load_saved_document_info():
    if not DOCUMENTS_INFO.exists():
        return {}

    try:
        data = json.loads(
            DOCUMENTS_INFO.read_text(encoding="utf-8")
        )
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_document_info(info):
    temp_file = DOCUMENTS_INFO.with_suffix(".tmp")
    temp_file.write_text(
        json.dumps(info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_file.replace(DOCUMENTS_INFO)


def load_saved_uploaded_store():
    index_file = VECTORSTORE_DIR / "index.faiss"
    pickle_file = VECTORSTORE_DIR / "index.pkl"

    if not index_file.exists() or not pickle_file.exists():
        return None

    try:
        return FAISS.load_local(
            str(VECTORSTORE_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception as error:
        st.warning(f"Saved document index could not be loaded: {error}")
        return None


def save_uploaded_store(vector_store):
    vector_store.save_local(str(VECTORSTORE_DIR))


def calculate_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


# --------------------------------------------------
# 4. PERSISTENT CHAT HISTORY
# --------------------------------------------------

def load_chat_sessions():
    if not CHAT_HISTORY_FILE.exists():
        return {}

    try:
        data = json.loads(
            CHAT_HISTORY_FILE.read_text(encoding="utf-8")
        )
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_chat_sessions(sessions):
    CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    temp_file = CHAT_HISTORY_FILE.with_suffix(".tmp")
    temp_file.write_text(
        json.dumps(sessions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_file.replace(CHAT_HISTORY_FILE)


def create_chat():
    chat_id = str(uuid.uuid4())

    return chat_id, {
        "title": "New conversation",
        "messages": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


# --------------------------------------------------
# 5. SESSION STATE
# --------------------------------------------------

if "uploaded_vector_store" not in st.session_state:
    st.session_state.uploaded_vector_store = load_saved_uploaded_store()

if "document_info" not in st.session_state:
    st.session_state.document_info = load_saved_document_info()

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_chat_sessions()

if not st.session_state.chat_sessions:
    chat_id, chat_data = create_chat()
    st.session_state.chat_sessions[chat_id] = chat_data
    st.session_state.active_chat_id = chat_id
    save_chat_sessions(st.session_state.chat_sessions)

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = next(
        iter(st.session_state.chat_sessions)
    )

if st.session_state.active_chat_id not in st.session_state.chat_sessions:
    st.session_state.active_chat_id = next(
        iter(st.session_state.chat_sessions)
    )

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

st.session_state.chat_history = st.session_state.chat_sessions[
    st.session_state.active_chat_id
]["messages"]


# --------------------------------------------------
# 6. SIDEBAR: CHAT HISTORY
# --------------------------------------------------

with st.sidebar:
    st.title("📚 Document Intelligence")
    st.caption("AI-powered PDF knowledge assistant")

    st.divider()
    st.subheader("💬 Chat History")

    if st.button("➕ New Chat", use_container_width=True):
        chat_id, chat_data = create_chat()
        st.session_state.chat_sessions[chat_id] = chat_data
        st.session_state.active_chat_id = chat_id
        st.session_state.chat_history = chat_data["messages"]
        save_chat_sessions(st.session_state.chat_sessions)
        st.rerun()

    chat_ids = list(st.session_state.chat_sessions.keys())

    selected_chat = st.selectbox(
        "Open a conversation",
        options=chat_ids,
        index=chat_ids.index(st.session_state.active_chat_id),
        format_func=lambda cid: st.session_state.chat_sessions[cid].get(
            "title", "Untitled conversation"
        ),
        key="selected_chat_id",
    )

    if selected_chat != st.session_state.active_chat_id:
        st.session_state.active_chat_id = selected_chat
        st.session_state.chat_history = (
            st.session_state.chat_sessions[selected_chat]["messages"]
        )
        st.rerun()

    if st.button("🗑️ Delete Current Chat", use_container_width=True):
        del st.session_state.chat_sessions[
            st.session_state.active_chat_id
        ]

        if not st.session_state.chat_sessions:
            new_id, new_chat = create_chat()
            st.session_state.chat_sessions[new_id] = new_chat

        st.session_state.active_chat_id = next(
            iter(st.session_state.chat_sessions)
        )
        st.session_state.chat_history = (
            st.session_state.chat_sessions[
                st.session_state.active_chat_id
            ]["messages"]
        )
        save_chat_sessions(st.session_state.chat_sessions)
        st.rerun()

    st.divider()
    st.subheader("📂 Knowledge Base")

    st.metric("Indexed PDFs", len(st.session_state.document_info))

    if st.session_state.document_info:
        with st.expander("View indexed documents"):
            for filename in sorted(st.session_state.document_info):
                st.write(f"📄 {filename}")

    st.caption("Conversations and indexed documents are saved locally.")


# --------------------------------------------------
# 7. MAIN HEADER
# --------------------------------------------------

st.title("Enterprise Document Intelligence Platform")

st.write(
    "Upload academic PDFs, ask questions about their content, "
    "and get answers with page-level source references."
)

st.divider()


# --------------------------------------------------
# 8. PDF UPLOAD AND INDEXING
# --------------------------------------------------

st.subheader("📂 Add Documents")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Previously indexed documents are preserved.",
)

if st.button(
    "⚙️ Process PDFs",
    type="primary",
    use_container_width=True,
):
    if not uploaded_files:
        st.warning("Please select at least one PDF first.")
    else:
        current_store = st.session_state.uploaded_vector_store
        current_info = dict(st.session_state.document_info)

        new_info = {}
        all_new_chunks = []
        errors = []
        skipped = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            add_start_index=True,
        )

        progress = st.progress(0)
        status = st.empty()

        for index, uploaded_file in enumerate(uploaded_files):
            filename = Path(uploaded_file.name).name
            file_bytes = uploaded_file.getvalue()
            file_hash = calculate_file_hash(file_bytes)

            status.write(f"Processing: {filename}")

            existing_hashes = {
                details.get("sha256")
                for details in current_info.values()
            }
            new_hashes = {
                details.get("sha256")
                for details in new_info.values()
            }

            if file_hash in existing_hashes or file_hash in new_hashes:
                skipped.append(filename)
                progress.progress((index + 1) / len(uploaded_files))
                continue

            try:
                saved_path = UPLOAD_DIR / filename
                saved_path.write_bytes(file_bytes)

                documents = load_pdf_documents(str(saved_path))

                if not documents:
                    errors.append(
                        f"{filename}: No readable text was found."
                    )
                    progress.progress((index + 1) / len(uploaded_files))
                    continue

                for document in documents:
                    document.metadata["file_name"] = filename
                    document.metadata["source"] = filename
                    document.metadata["sha256"] = file_hash

                    if document.metadata.get("page_number") is None:
                        page = document.metadata.get("page")
                        if page is not None:
                            document.metadata["page_number"] = int(page) + 1

                chunks = splitter.split_documents(documents)

                if not chunks:
                    errors.append(f"{filename}: No text chunks were created.")
                    progress.progress((index + 1) / len(uploaded_files))
                    continue

                all_new_chunks.extend(chunks)

                new_info[filename] = {
                    "sha256": file_hash,
                    "chunks": len(chunks),
                    "uploaded_at": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                }

            except Exception as error:
                errors.append(f"{filename}: {error}")

            progress.progress((index + 1) / len(uploaded_files))

        if all_new_chunks:
            try:
                if current_store is None:
                    current_store = FAISS.from_documents(
                        all_new_chunks,
                        get_embeddings(),
                    )
                else:
                    current_store.add_documents(all_new_chunks)

                save_uploaded_store(current_store)

                current_info.update(new_info)
                save_document_info(current_info)

                st.session_state.uploaded_vector_store = current_store
                st.session_state.document_info = current_info

                st.success(
                    f"Successfully indexed {len(new_info)} new PDF(s). "
                    f"Knowledge base now contains {len(current_info)} PDF(s)."
                )

            except Exception as error:
                st.error(f"Could not save the document index: {error}")
                st.exception(error)

        elif not errors and skipped:
            st.info("These PDFs were already indexed. No duplicate chunks added.")

        for filename in skipped:
            st.caption(f"Skipped duplicate: {filename}")

        for message in errors:
            st.error(message)

        progress.empty()
        status.empty()

        if all_new_chunks:
            st.rerun()


# --------------------------------------------------
# 9. ACTIVE KNOWLEDGE BASE
# --------------------------------------------------

vector_store = st.session_state.uploaded_vector_store

if vector_store is not None:
    st.success(
        f"Knowledge base ready — "
        f"{len(st.session_state.document_info)} PDF(s) indexed."
    )
else:
    st.info("Upload and process a PDF to start asking questions.")


# --------------------------------------------------
# 10. SUGGESTED QUESTIONS
# --------------------------------------------------

st.subheader("💡 Try asking")

suggestions = [
    "Summarize the main concepts.",
    "Explain the important definitions.",
    "What are the key topics in this document?",
]

columns = st.columns(len(suggestions))

for column, suggestion in zip(columns, suggestions):
    with column:
        if st.button(suggestion, use_container_width=True):
            st.session_state.pending_question = suggestion
            st.rerun()


# --------------------------------------------------
# 11. DISPLAY ACTIVE CHAT
# --------------------------------------------------

st.subheader("💬 Ask Your Documents")

for item in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(item["question"])

    with st.chat_message("assistant"):
        st.markdown(item["answer"])

        sources = item.get("sources", [])

        if sources:
            with st.expander("📚 View Sources"):
                for source in sources:
                    filename = source.get(
                        "filename",
                        source.get(
                            "file",
                            source.get("source", "Unknown file"),
                        ),
                    )
                    page = source.get("page", "N/A")
                    st.markdown(f"- **{filename}** — Page {page}")


# --------------------------------------------------
# 12. QUESTION ANSWERING
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your PDFs...",
    key="main_question_input",
)

if not question:
    question = st.session_state.pending_question

st.session_state.pending_question = None

if question and question.strip():
    question = question.strip()

    if vector_store is None:
        st.warning("Please upload and process a PDF first.")
    else:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    answer, sources = answer_question(
                        vector_store,
                        question,
                    )

                    if not answer:
                        answer = (
                            "I could not generate an answer. "
                            "Please try rephrasing your question."
                        )

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 View Sources"):
                            for source in sources:
                                filename = source.get(
                                    "filename",
                                    source.get(
                                        "file",
                                        source.get("source", "Unknown file"),
                                    ),
                                )
                                page = source.get("page", "N/A")
                                st.markdown(
                                    f"- **{filename}** — Page {page}"
                                )
                    else:
                        st.caption("No source references were returned.")

                    active_chat = st.session_state.chat_sessions[
                        st.session_state.active_chat_id
                    ]

                    active_chat["messages"].append({
                        "question": question,
                        "answer": answer,
                        "sources": sources or [],
                    })

                    if active_chat["title"] == "New conversation":
                        active_chat["title"] = question[:50]

                    save_chat_sessions(st.session_state.chat_sessions)

                except Exception as error:
                    st.error(f"Could not generate answer: {error}")
