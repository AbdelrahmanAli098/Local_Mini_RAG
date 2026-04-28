import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import (
    extract_text_from_pdf,
    init_spacy,
    normalize_text_chunks,
    split_into_chunks,
    split_into_sentences,
)


# Build the document index from a PDF.
# This function reads the PDF, splits text into sentences, groups sentences into chunks,
# filters out very small chunks, and returns metadata for each chunk.
def build_text_chunks(
    pdf_path: str,
    chunk_size: int = 10,
    min_token_count: int = 30,
    nlp: Optional[Any] = None,) -> List[Dict[str, Any]]:
    if nlp is None:
        nlp = init_spacy()

    pages = extract_text_from_pdf(pdf_path)
    chunks: List[Dict[str, Any]] = []

    for page in pages:
        # Split the page text into individual sentences
        sentences = split_into_sentences(page["text"], nlp)
        # Group sentences into fixed-size chunks for embedding
        sentence_chunks = split_into_chunks(sentences, chunk_size=chunk_size)

        for sentence_chunk in sentence_chunks:
            chunk_text = normalize_text_chunks(sentence_chunk)
            token_count = len(chunk_text) / 4.0
            if token_count < min_token_count:
                continue

            chunks.append(
                {
                    "page_number": page["page_number"],
                    "sentence_chunk": chunk_text,
                    "chunk_char_count": len(chunk_text),
                    "chunk_word_count": len(chunk_text.split()),
                    "chunk_token_count": token_count,
                }
            )

    return chunks


def encode_chunks(
    chunks: List[Dict[str, Any]],
    model_name: str = "all-mpnet-base-v2",
    device: Optional[str] = None,
    batch_size: int = 16,) -> Tuple[List[Dict[str, Any]], torch.Tensor]:

    # Load the sentence embedding model and encode each chunk text.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name, device=device)
    texts = [chunk["sentence_chunk"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    for chunk, embedding in zip(chunks, embeddings.cpu().numpy()):
        chunk["embedding"] = embedding.tolist()

    return chunks, embeddings


# Persist chunk metadata and embeddings to disk for later reuse.
def save_chunks_to_csv(chunks: List[Dict[str, Any]], csv_path: str) -> None:
    df = pd.DataFrame(chunks)
    if "embedding" in df.columns:
        df["embedding"] = df["embedding"].apply(json.dumps)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def _parse_embedding_value(value: Any) -> np.ndarray:
    embedding_text = str(value).strip()
    if embedding_text.startswith("[") and embedding_text.endswith("]"):
        try:
            return np.array(json.loads(embedding_text), dtype=np.float32)
        except json.JSONDecodeError:
            cleaned = embedding_text[1:-1].replace("\n", " ").strip()
            return np.fromstring(cleaned, sep=" ", dtype=np.float32)
    raise ValueError("Unsupported embedding format")


def load_chunks_from_csv(csv_path: str, device: Optional[str] = None) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    # Load chunk metadata and embeddings from the saved CSV index.
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")

    if "embedding" not in df.columns:
        raise ValueError("CSV must contain an embedding column.")

    embeddings = torch.stack(
        [torch.tensor(_parse_embedding_value(record["embedding"]), dtype=torch.float32) for record in records]
    )
    if device is not None:
        embeddings = embeddings.to(device)
    return records, embeddings


def load_embedding_model(model_name: str = "all-mpnet-base-v2", device: Optional[str] = None,) -> SentenceTransformer:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def load_llm_model(
    model_id: str = "google/gemma-2b-it",
    device: Optional[str] = None,
    load_in_4bit: bool = True,
    compute_dtype: Optional[str] = "auto",) -> Tuple[Any, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=torch.float16 if compute_dtype == "auto" else getattr(torch, compute_dtype),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.float16,
    )
    return model, tokenizer


def retrieve_relevant_chunks(
    query: str,
    embeddings: torch.Tensor,
    chunks: List[Dict[str, Any]],
    embedding_model: SentenceTransformer,
    top_k: int = 5,
    device: Optional[str] = None) -> List[Dict[str, Any]]:
    # Retrieve the top-k text chunks that are most similar to the query.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = embeddings.to(device)
    query_embedding = embedding_model.encode(query, convert_to_tensor=True, device=device)
    scores = util.dot_score(query_embedding, embeddings)[0]
    top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)))

    results: List[Dict[str, Any]] = []
    for score, index in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist()):
        item = chunks[index].copy()
        # Always strip embedding metadata before returning items.
        item.pop("embedding", None)
        item["score"] = float(score)
        results.append(item)

    return results


def format_prompt(query: str, context_items: List[Dict[str, Any]]) -> str:
    context_text = "\n\n".join(
        [f"Context {idx + 1}: {item['sentence_chunk']}" for idx, item in enumerate(context_items)]
    )
    return (
        "Use the context passages below to answer the user query. "
        "Do not invent information that is not present in the context.\n\n"
        f"{context_text}\n\n"
        f"User query: {query}\n\n"
        "Answer:"
    )


def generate_answer(
    prompt: str,
    llm_model: Any,
    tokenizer: Any,
    device: Optional[str] = None,
    max_new_tokens: int = 512, do_sample: bool = True,) -> str:

    # Generate a final answer from the LLM using the constructed prompt.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Full RAG pipeline: retrieve relevant chunks, build prompt, and generate answer.
def ask_with_rag(
    query: str,
    chunks: List[Dict[str, Any]],
    embeddings: torch.Tensor,
    embedding_model: SentenceTransformer,
    llm_model: Any,
    tokenizer: Any,
    top_k: int = 5,
    device: Optional[str] = None,) -> Tuple[str, List[Dict[str, Any]]]:
    context_items = retrieve_relevant_chunks(
        query=query,
        embeddings=embeddings,
        chunks=chunks,
        embedding_model=embedding_model,
        top_k=top_k,
        device=device,
    )
    prompt = format_prompt(query, context_items)
    answer = generate_answer(
        prompt=prompt,
        llm_model=llm_model,
        tokenizer=tokenizer,
        device=device,
    )
    return answer, context_items
