# PDF Document Q&A – RAG Application

## Overview

A Python-based Retrieval-Augmented Generation (RAG) application that answers questions using only information contained in the supplied PDF documents. The application extracts PDF text, creates embeddings, stores them in Qdrant, retrieves relevant content for a user query, and uses an OpenRouter free LLM to generate a grounded answer with document and page-level citations.

## Architecture

```text
PDF Documents
      ↓
PDF Text Extraction
      ↓
Text Chunking
      ↓
Embedding Model
      ↓
Qdrant Vector Database
      ↓
User Question
      ↓
Query Embedding
      ↓
Similarity Search
      ↓
Relevant Chunks
      ↓
OpenRouter LLM
      ↓
Answer + Citations
```

The Streamlit interface provides the user-facing application.

## Libraries / Technologies

* **Python 3.11+**
* **pypdf** – PDF text extraction
* **Sentence Transformers (all-MiniLM-L6-v2)** – text embeddings
* **Qdrant** – vector database
* **OpenAI Python SDK** – OpenRouter API integration
* **python-dotenv** – environment variable management
* **Streamlit** – application interface

## Embedding Model

The application uses a Sentence Transformers embedding model to convert document chunks and user questions into numerical vectors. These vectors are stored in Qdrant and compared using vector similarity search.

## Assumptions

* The supplied PDFs contain machine-readable text.
* Answers must be based only on the supplied documents.
* If relevant information cannot be retrieved, the application returns:
  **"The information is not available in the supplied documents."**
* Citations contain the document name, page number, and retrieved supporting text.
* The OpenRouter API key is stored locally in a `.env` file and is not committed to GitHub.

## How to Run

### 1. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the API key

Create a `.env` file:

```text
OPENROUTER_API_KEY=your_api_key_here
```

### 4. Start Qdrant

Ensure the local Qdrant instance is running on port `6333`.

### 5. Ingest the PDFs

```bash
python ingest.py
```

### 6. Start the application

```bash
streamlit run app.py
```

The application will be available at the local Streamlit URL shown in the terminal.
