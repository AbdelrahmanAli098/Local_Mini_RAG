# Local Mini RAG

A small PDF-based Retrieval-Augmented Generation (RAG) project built with FastAPI. This project extracts text from a nutrition PDF, creates text chunks, embeds them with `sentence-transformers`, stores chunk metadata and embeddings in a CSV index, and answers user questions using a text generation model.

## Project Overview

- `app/main.py` - FastAPI application exposing endpoints for building the index, searching similar chunks, and answering questions with RAG.
- `app/pipeline.py` - RAG pipeline logic: chunk creation, embedding encoding, CSV save/load, similarity retrieval, prompt formatting, and answer generation.
- `app/utils.py` - PDF extraction and text preprocessing utilities.
- `requirements.txt` - Python dependencies.
- `dockerfile` - currently an empty placeholder; the project can be dockerized if you add a proper Dockerfile.
- `data/` - expected folder for `human-nutrition-text.pdf` and generated `text_chunk_embeddings.csv`.
- `notebook/` - includes original notebook exploration and example pipeline code.

## Features

- Extracts text from PDF pages using `PyMuPDF` (`fitz`).
- Splits PDF text into sentence chunks with `spaCy`.
- Creates chunk embeddings with `SentenceTransformer` (`all-mpnet-base-v2`).
- Persists chunk metadata and embeddings to a CSV file.
- Searches the index using dot-product similarity.
- Uses a Hugging Face `transformers` model for answer generation.
- Returns answers and relevant context without exposing raw embedding vectors in the API response.

## Installation

1. Create and activate a Python virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Download or place your PDF in `data/human-nutrition-text.pdf`.

## Running Locally

From the project root (`d:\Projects\Local_Mini_RAG`):

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open `http://127.0.0.1:8000/docs` for the FastAPI Swagger UI.

## API Endpoints

### `POST /build-index`
Builds the embedding index from the PDF.

Request body example:

```json
{
  "pdf_path": "data/human-nutrition-text.pdf",
  "chunk_size": 10,
  "min_token_count": 30
}
```

### `POST /search`
Searches the loaded index for top matching chunks.

Request body example:

```json
{
  "question": "What foods are high in protein?",
  "top_k": 5
}
```

### `POST /ask-question`
Runs RAG: retrieves context and generates an answer from the LLM.

Request body example:

```json
{
  "question": "What foods are high in protein?",
  "top_k": 5
}
```

Response example:

```json
{
  "answer": "...",
  "context": [
    {
      "page_number": 411,
      "sentence_chunk": "...",
      "chunk_char_count": 432,
      "chunk_word_count": 70,
      "chunk_token_count": 108.0,
      "score": 0.77
    }
  ]
}
```

Note: the `context` items are sanitized so `embedding` arrays are not returned.

### `GET /health`
Checks app health and loaded model/index status.

## Notes

- `app/main.py` loads the embedding and language models during startup.
- The index is stored in `data/text_chunk_embeddings.csv`.
- `app/utils.py` uses an offset page number starting at 41 to match PDF page metadata.
- GPU support is available if PyTorch detects CUDA; otherwise the app runs on CPU.

## Docker

This project can be run via Docker (handy if you don’t want to set up Python locally).

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
