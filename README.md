# Local Mini RAG

A small PDF-based Retrieval-Augmented Generation (RAG) project built with FastAPI. Upload PDFs dynamically, extract text, create sentence chunks, embed them using `sentence-transformers`, store indices in-memory per document, and answer user questions using a local text generation model.

## Project Overview

- `app/main.py` - FastAPI application exposing endpoints for uploading PDFs, searching similar chunks, and answering questions with RAG.
- `app/pipeline.py` - RAG pipeline logic: chunk creation, embedding encoding, similarity retrieval, prompt formatting, and answer generation.
- `app/utils.py` - PDF extraction and text preprocessing utilities.
- `requirements.txt` - Python dependencies.
- `dockerfile` - Placeholder for dockerization.
- `data/` - Folder containing example data like `text_chunk_embeddings.csv` (legacy or for reference).
- `notebook/` - Jupyter notebooks for exploration:
  - `Preprocessing.ipynb` - Data preprocessing, PDF text extraction, and chunking.
  - `Rag.ipynb` - Demonstration of the RAG pipeline.

## Features

- Extracts text from uploaded PDF pages using `PyMuPDF` (`fitz`).
- Splits PDF text into sentence chunks with `spaCy`.
- Creates chunk embeddings with `SentenceTransformer` (`all-mpnet-base-v2`).
- Stores chunk metadata and embeddings in-memory per uploaded document.
- Searches the index using dot-product similarity.
- Uses a Hugging Face `transformers` model for answer generation.
- Returns answers and relevant context without exposing raw embedding vectors.

## Installation

1. Create and activate a Python virtual environment.

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

## Running Locally

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open `http://127.0.0.1:8000/docs` for the FastAPI Swagger UI.

## API Endpoints

### `POST /documents/upload`
Upload a PDF, process it into chunks and embeddings, and store in memory. Returns a `doc_id` for subsequent operations.

Request: Multipart form data with `file` (PDF), `chunk_size` (default 10), `min_token_count` (default 30).

Response example:

```json
{
  "doc_id": "uuid-string",
  "filename": "example.pdf",
  "chunks": 42
}
```

### `POST /documents/{doc_id}/search`
Search the uploaded document's index for top matching chunks.

Request body:

```json
{
  "question": "What foods are high in protein?",
  "top_k": 5
}
```

Response example:

```json
{
  "doc_id": "uuid-string",
  "results": [
    {
      "page_number": 411,
      "sentence_chunk": "...",
      "chunk_char_count": 123,
      "chunk_word_count": 20,
      "chunk_token_count": 25
    }
  ]
}
```

### `POST /documents/{doc_id}/ask`
Run RAG on the uploaded document: retrieve context and generate an answer.

Request body:

```json
{
  "question": "What foods are high in protein?",
  "top_k": 5
}
```

Response example:

```json
{
  "doc_id": "uuid-string",
  "answer": "Foods high in protein include...",
  "context": [
    {
      "page_number": 411,
      "sentence_chunk": "...",
      "chunk_char_count": 123,
      "chunk_word_count": 20,
      "chunk_token_count": 25
    }
  ]
}
```

### `DELETE /documents/{doc_id}`
Delete the uploaded document's index from memory.

Response:

```json
{
  "deleted": "uuid-string"
}
```

### `GET /health`
Health check endpoint.

Response example:

```json
{
  "status": "healthy",
  "chunks_loaded": 0,
  "documents_in_memory": 1,
  "device": "cuda",
  "embedding_model": true,
  "llm_model": true
}
```

## Notebooks

- **Preprocessing.ipynb**: Demonstrates PDF text extraction, chunking, and embedding creation. Useful for understanding the preprocessing pipeline.
- **Rag.ipynb**: Shows the full RAG process, including retrieval and generation, with examples.

To run the notebooks, ensure you have Jupyter installed and the virtual environment activated.

## Docker

The `dockerfile` is a placeholder. To dockerize the application, add a proper Dockerfile with Python base image, copy files, install requirements, and expose port 8000.
      "chunk_char_count": 432,
      "chunk_word_count": 70,
      "chunk_token_count": 108.0,
      "score": 0.77
    }
  ]
}
```

Note: the `context` items are sanitized so `embedding` arrays are not returned.

### `POST /documents/upload`
Upload a PDF and build an in-memory index (extract → chunk → embed → store).

Use Swagger UI (`/docs`) and choose **multipart/form-data**:

- `file`: your PDF
- `chunk_size`: (optional) default `10`
- `min_token_count`: (optional) default `30`

Response example:

```json
{
  "doc_id": "c3c6b4df-0c3a-4a9f-9a1f-9b31b8b6f5d2",
  "filename": "my.pdf",
  "chunks": 123
}
```

### `POST /documents/{doc_id}/search`
Search within the uploaded document’s in-memory index.

### `POST /documents/{doc_id}/ask`
Run RAG against the uploaded document’s in-memory index.

### `DELETE /documents/{doc_id}`
Delete an uploaded document index and remove its temp PDF.

### `GET /health`
Checks app health and loaded model/index status.

## Notes

- `app/main.py` loads the embedding and language models during startup.
- The index is stored in `data/text_chunk_embeddings.csv`.
- Page numbers are now 1-based for any PDF by default.
- GPU support is available if PyTorch detects CUDA; otherwise the app runs on CPU.

## Docker

This project can be run via Docker.

### Build

From the repo root:

```powershell
docker build -f dockerfile -t local-mini-rag .
```

### Run

Start the API and expose it on `http://127.0.0.1:8000`:

```powershell
docker run --rm -p 8000:8000 local-mini-rag
```

Open the Swagger UI at `http://127.0.0.1:8000/docs`.

### Use local `data/` (recommended)

To let the container read `data/human-nutrition-text.pdf` and write `data/text_chunk_embeddings.csv` back to your machine, mount the `data/` folder:

```powershell
docker run --rm -p 8000:8000 -v "${PWD}\data:/code/data" local-mini-rag
```

Then use `data/human-nutrition-text.pdf` as the `pdf_path` when calling `POST /build-index`.

## Recommended Workflow

1. Start the FastAPI app.
2. Call `/build-index` to generate the CSV index from the PDF.
3. Call `/search` to inspect retrieval results.
4. Call `/ask-question` to get an answer based on PDF context.

## Troubleshooting

- If `/build-index` fails, confirm `data/human-nutrition-text.pdf` exists.
- If models fail to load, verify the Python environment and installed packages.
- Use `uvicorn app.main:app` from the project root so imports resolve correctly.