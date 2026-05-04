from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from .pipeline import (
    ask_with_rag,
    build_text_chunks,
    load_embedding_model,
    load_llm_model,
    retrieve_relevant_chunks,
)

# Global state for models and data
embedding_model = None
llm_model = None
tokenizer = None
chunks = []
embeddings = None
device = None

# In-memory, per-upload indices: doc_id -> {chunks, embeddings, filename, tmp_path}
doc_indices: Dict[str, dict] = {}


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

    yield

    # Cleanup if needed
    pass


app = FastAPI(lifespan=lifespan)
@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(10),
    min_token_count: int = Form(30),
):
    """
    Upload a PDF, extract text dynamically, chunk it, embed it, and store the index in memory.
    Returns a doc_id that can be used for subsequent /documents/{doc_id}/search and /documents/{doc_id}/ask calls.
    """
    if file.content_type not in {"application/pdf", "application/x-pdf", "application/acrobat", "applications/vnd.pdf"}:
        # Don't hard-fail on weird content-types, but block obvious non-pdf uploads
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    doc_id = str(uuid.uuid4())
    tmp_dir = Path(tempfile.gettempdir()) / "local-mini-rag"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{doc_id}.pdf"

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    tmp_path.write_bytes(data)

    # Build chunks from the uploaded PDF
    local_chunks = build_text_chunks(
        pdf_path=str(tmp_path),
        chunk_size=chunk_size,
        min_token_count=min_token_count,
    )
    if not local_chunks:
        raise HTTPException(
            status_code=400,
            detail="No chunks produced from this PDF (try lowering min_token_count or check PDF text extractability).",
        )

    # Embed chunks with the already-loaded embedding model
    texts = [c["sentence_chunk"] for c in local_chunks]
    local_embeddings = embedding_model.encode(
        texts, batch_size=16, convert_to_tensor=True, show_progress_bar=True
    )
    if device is not None:
        local_embeddings = local_embeddings.to(device)
    for chunk, emb in zip(local_chunks, local_embeddings.detach().cpu().numpy()):
        chunk["embedding"] = emb.tolist()

    doc_indices[doc_id] = {
        "chunks": local_chunks,
        "embeddings": local_embeddings,
        "filename": file.filename,
        "tmp_path": str(tmp_path),
    }

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks": len(local_chunks),
    }


class DocSearchRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


def _sanitize_items(items: List[dict]) -> List[dict]:
    return [{k: v for k, v in item.items() if k != "embedding"} for item in items]


@app.post("/documents/{doc_id}/search")
def search_document(doc_id: str, request: DocSearchRequest):
    index = doc_indices.get(doc_id)
    if index is None:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Upload a document first.")

    results = retrieve_relevant_chunks(
        query=request.question,
        embeddings=index["embeddings"],
        chunks=index["chunks"],
        embedding_model=embedding_model,
        top_k=request.top_k,
        device=device,
    )
    return {"doc_id": doc_id, "results": _sanitize_items(results)}


@app.post("/documents/{doc_id}/ask")
def ask_document(doc_id: str, request: DocSearchRequest):
    index = doc_indices.get(doc_id)
    if index is None:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Upload a document first.")

    answer, context_items = ask_with_rag(
        query=request.question,
        chunks=index["chunks"],
        embeddings=index["embeddings"],
        embedding_model=embedding_model,
        llm_model=llm_model,
        tokenizer=tokenizer,
        top_k=request.top_k,
        device=device,
    )
    return {"doc_id": doc_id, "answer": answer, "context": _sanitize_items(context_items)}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    index = doc_indices.pop(doc_id, None)
    if index is None:
        raise HTTPException(status_code=404, detail="Unknown doc_id.")
    # Best-effort cleanup of temp file
    try:
        Path(index["tmp_path"]).unlink(missing_ok=True)
    except Exception:
        pass
    return {"deleted": doc_id}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "chunks_loaded": len(chunks),
        "documents_in_memory": len(doc_indices),
        "device": device,
        "embedding_model": embedding_model is not None,
        "llm_model": llm_model is not None,
    }