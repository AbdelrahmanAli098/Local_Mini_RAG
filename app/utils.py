import fitz
import re
from typing import List
from spacy.lang.en import English

def text_formatter(text: str) -> str:
    # Normalize raw text by removing newlines and collapsing spaces
    return " ".join(text.split()).strip()

def init_spacy():
    nlp = English()
    nlp.add_pipe("sentencizer")
    return nlp

def split_into_sentences(text: str, nlp: English) -> List[str]:
    doc = nlp(text)
    # Create a list of sentences by iterating through the document's sentences and stripping any leading/trailing whitespace
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def split_into_chunks(sentences: List[str], chunk_size: int =10) -> list[list[str]]:
    chunks = []
    if chunk_size < 1:
        raise ValueError("slice_size must be >= 1")
    # Loop through the sentences in steps of chunk_size and create chunks
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def normalize_text_chunks(sentence_chunk: List[str]) -> List[str]:
    chunk_text = " ".join(sentence_chunk)
    # Remove extra spaces and newlines from the chunk text
    chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
    # Add a space after periods followed by uppercase letters to ensure proper sentence separation 
    chunk_text = re.sub(r"\.([A-Z])", r". \1", chunk_text)

    return chunk_text

def extract_text_from_pdf(pdf_path: str) -> str:

    doc = fitz.open(pdf_path)
    pages = []
    # Iterate through each page in the PDF document and extract the text, clean it, and store it in a list of dictionaries
    for page_number, page in enumerate(doc):
        raw_text = page.get_text()
        cleaned_text = text_formatter(raw_text)
        pages.append(
            {
                "page_number": page_number + 41,
                "text": cleaned_text,
                "page_char_count": len(cleaned_text),
                "page_word_count": len(cleaned_text.split())
            }
        ) 
    
    return pages
