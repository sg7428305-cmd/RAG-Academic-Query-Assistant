# RAG Academic Query Assistant

An AI-powered PDF knowledge assistant that answers academic questions using information retrieved from uploaded PDF documents.

## Project Overview

The RAG Academic Query Assistant uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from PDFs and generate answers grounded in the retrieved content.

It also displays source references and handles questions when relevant information is not available in the uploaded documents.

## Features

- Upload and process PDF documents
- Ask questions in natural language
- Retrieve relevant document chunks using FAISS
- Generate answers using a Groq-hosted language model
- Display source PDF names and page references
- Maintain chat history
- Evaluate retrieval performance
- Reject questions when relevant document information is not found

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face Embeddings
- Groq API
- PyMuPDF / PDF processing tools

## Project Structure

```text
RAG-Academic-Query-Assistant/
├── app.py
├── evaluate_retrieval.py
├── evaluation_dashboard.py
├── evaluation_report.json
├── src/
│   ├── document_loader.py
│   ├── vector_store.py
│   ├── search_documents.py
│   └── ask_question.py
├── data/
│   ├── documents/
│   └── vectorstore/
├── .env
└── README.md
```

## Installation

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the project dependencies.
4. Add your Groq API key to a `.env` file.

Example `.env` entry:

```text
GROQ_API_KEY=your_api_key_here
```

Never commit your real API key to GitHub.

## Run the Application

```powershell
python -m streamlit run app.py
```

## Retrieval Evaluation

Run the retrieval evaluation script:

```powershell
python evaluate_retrieval.py
```

The evaluation report is saved to `evaluation_report.json`.

## Example Use Cases

- Ask questions about DBMS notes
- Understand Machine Learning concepts from uploaded PDFs
- Find relevant pages in academic documents
- Build a searchable knowledge base from study materials

## Future Improvements

- Improve retrieval accuracy with more evaluation queries
- Add reranking for better document relevance
- Support additional document formats
- Add automated tests and deployment
- Improve document-level access and chat management

## Author

BCA Artificial Intelligence and Machine Learning Student
