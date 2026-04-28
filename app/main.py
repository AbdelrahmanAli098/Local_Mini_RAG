from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import (
    ask_with_rag,
    build_text_chunks,
    encode_chunks,
    load_chunks_from_csv,
    load_embedding_model,
    load_llm_model,
    retrieve_relevant_chunks,
    save_chunks_to_csv,
)

# Global state for models and data
embedding_model = None
llm_model = None
tokenizer = None
chunks = []
embeddings = None
device = None

# Paths
PDF_PATH = Path(__file__).parent.parent / "data" / "human-nutrition-text.pdf"
CSV_PATH = Path(__file__).parent.parent / "data" / "text_chunk_embeddings.csv"


class Question(BaseModel):
    question: str


class BuildIndexRequest(BaseModel):
    pdf_path: Optional[str] = None
    chunk_size: Optional[int] = 10
    min_token_count: Optional[int] = 30


class SearchRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, llm_model, tokenizer, chunks, embeddings, device

    # Determine device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load embedding model
    embedding_model = load_embedding_model(device=device)

    # Load LLM and tokenizer
    llm_model, tokenizer = load_llm_model(device=device)

    # Try to load existing chunks and embeddings
    if CSV_PATH.exists():
        try:
            chunks, embeddings = load_chunks_from_csv(str(CSV_PATH), device=device)
            print(f"Loaded {len(chunks)} chunks from {CSV_PATH}")
        except Exception as e:
            print(f"Failed to load chunks: {e}")
            chunks = []
            embeddings = None
    else:
        print(f"No existing index found at {CSV_PATH}")

    yield

    # Cleanup if needed
    pass


app = FastAPI(lifespan=lifespan)


@app.post("/build-index")
def build_index(request: BuildIndexRequest):
    global chunks, embeddings

    pdf_path = request.pdf_path or str(PDF_PATH)
    if not Path(pdf_path).exists():
        raise HTTPException(status_code=404, detail="PDF file not found")

    # Build chunks
    chunks = build_text_chunks(
        pdf_path=pdf_path,
        chunk_size=request.chunk_size,
        min_token_count=request.min_token_count,
    )

    # Encode chunks
    chunks, embeddings = encode_chunks(chunks, device=device)

    # Save to CSV
    save_chunks_to_csv(chunks, str(CSV_PATH))

    return {"message": f"Built index with {len(chunks)} chunks", "csv_path": str(CSV_PATH)}


def _sanitize_items(items: List[dict]) -> List[dict]:
    return [{k: v for k, v in item.items() if k != "embedding"} for item in items]


@app.post("/search")
def search(request: SearchRequest):
    if not chunks or embeddings is None:
        raise HTTPException(status_code=400, detail="Index not loaded. Call /build-index first.")

    results = retrieve_relevant_chunks(
        query=request.question,
        embeddings=embeddings,
        chunks=chunks,
        embedding_model=embedding_model,
        top_k=request.top_k,
        device=device,
    )

    return {"results": _sanitize_items(results)}


@app.post("/ask-question")
def ask_question(request: AskRequest):
    if not chunks or embeddings is None:
        raise HTTPException(status_code=400, detail="Index not loaded. Call /build-index first.")

    answer, context_items = ask_with_rag(
        query=request.question,
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=embedding_model,
        llm_model=llm_model,
        tokenizer=tokenizer,
        top_k=request.top_k,
        device=device,
    )

    return {"answer": answer, "context": _sanitize_items(context_items)}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "chunks_loaded": len(chunks),
        "device": device,
        "embedding_model": embedding_model is not None,
        "llm_model": llm_model is not None,
    }