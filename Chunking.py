import os
from pathlib import Path
import pdfplumber
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from mistralai import Mistral
from bs4 import BeautifulSoup

# Download punkt tokenizer
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize

# Initialize Mistral client for agentic chunking
api_key = os.getenv("MISTRAL_API_KEY", "YOUR_API_KEY")
client = Mistral(api_key=api_key)


# ========== Fixed-size chunking ==========
def fixed_sized_chunking(text, chunk_size=500):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


# ========== Recursive chunking ==========
def recursive_chunking(text, separators, chunk_size=300):
    if len(text) <= chunk_size:
        return [text]
    current_sep = separators[0] if separators else None
    if current_sep and current_sep in text:
        parts = text.split(current_sep)
        chunks, current = [], ""
        for part in parts:
            if len(current) + len(part) + len(current_sep) <= chunk_size:
                current += part + current_sep
            else:
                if current:
                    chunks.append(current.strip())
                current = part + current_sep
        if current:
            chunks.append(current.strip())
        if len(separators) > 1:
            refined = []
            for c in chunks:
                if len(c) > chunk_size:
                    refined.extend(recursive_chunking(c, separators[1:], chunk_size))
                else:
                    refined.append(c)
            return refined
        return chunks
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


# ========== Document page-based chunking ==========
def chunk_by_page(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == ".pdf":
        return _chunk_pdf_by_page(file_path)
    elif ext in (".html", ".htm"):
        text = _load_html(file_path)
        # fallback to fixed or paragraph chunks if desired
        return fixed_sized_chunking(text, chunk_size=1500)
    elif ext == ".txt":
        return _chunk_txt_by_page(file_path)
    else:
        raise ValueError("Unsupported format: use .pdf, .txt or .html")


def _chunk_pdf_by_page(pdf_path):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            text = p.extract_text()
            if text:
                chunks.append(text.strip())
    return chunks


def _chunk_txt_by_page(txt_path, max_chars=1500):
    text = Path(txt_path).read_text(encoding="utf-8")
    chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    return [c.strip() for c in chunks if c.strip()]


# ========== HTML loader ==========
def _load_html(html_path):
    html = Path(html_path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


# ========== Sentence-based chunking ==========
def sentence_based_chunking(text, n_sentences=5):
    sentences = sent_tokenize(text)
    return [
        " ".join(sentences[i : i + n_sentences])
        for i in range(0, len(sentences), n_sentences)
    ]


# ========== Semantic chunking ==========
def semantic_chunking(text, model_name="all-MiniLM-L6-v2", threshold=0.7):
    sentences = sent_tokenize(text)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        if sim >= threshold:
            current.append(sentences[i])
        else:
            chunks.append(" ".join(current))
            current = [sentences[i]]
    if current:
        chunks.append(" ".join(current))
    return chunks


# ========== Agentic chunking ==========
def agentic_chunking(text, task_description):
    prompt = f"""Tu es un assistant intelligent. Tu dois découper le texte suivant en sections sémantiques utiles pour : {task_description}.
Texte :
{text}
Renvoie une liste de morceaux clairs et utiles pour cette tâche, séparés par deux retours à la ligne."""
    messages = [{"role": "user", "content": prompt}]
    resp = client.chat.complete(model="mistral-large-latest", messages=messages)
    raw = resp.choices[0].message.content
    return [ch.strip() for ch in raw.split("\n\n") if ch.strip()]


# ========== Unified chunking interface ==========
def chunk_document(source, method="fixed", **kwargs):
    """
    source: text string or file path (.pdf, .txt, .html)
    method: 'fixed','recursive','page','sentence','semantic','agentic'
    kwargs: method-specific args
    """
    text = source
    raw_html = None
    # If source is a file, load its raw and cleaned content
    if isinstance(source, (str, Path)) and os.path.isfile(str(source)):
        _, ext = os.path.splitext(str(source))
        ext = ext.lower()
        if method == "page":
            return chunk_by_page(str(source))
        if ext == ".pdf":
            text = "\n".join(_chunk_pdf_by_page(str(source)))
        elif ext in (".html", ".htm"):
            raw_html = Path(source).read_text(encoding="utf-8")
            text = _load_html(str(source))
        elif ext == ".txt":
            text = Path(source).read_text(encoding="utf-8")
        else:
            raise ValueError("Unsupported file type")
    # Precompute separators for recursive chunking
    if method == "recursive":
        if raw_html:
            # detect HTML separators first
            possible_html = ["</p>", "<br>", "</div>", "</h1>", "</h2>"]
            separators = [tag for tag in possible_html if tag in raw_html]
        else:
            possible = ["\n\n", "\f", "\n", ". "]
            separators = [s for s in possible if s in text]
        return recursive_chunking(
            raw_html if raw_html else text, separators, kwargs.get("chunk_size", 300)
        )
    # Dispatch other methods
    if method == "fixed":
        return fixed_sized_chunking(text, kwargs.get("chunk_size", 500))
    if method == "sentence":
        return sentence_based_chunking(text, kwargs.get("n_sentences", 5))
    if method == "semantic":
        return semantic_chunking(
            text,
            kwargs.get("model_name", "all-MiniLM-L6-v2"),
            kwargs.get("threshold", 0.7),
        )
    if method == "agentic":
        return agentic_chunking(
            text, kwargs.get("task_description", "traiter le document")
        )
    raise ValueError(f"Unknown method: {method}")


# Example usage
def example():
    path = "path/to/doc.pdf"
    for m in ["fixed", "recursive", "page", "sentence", "semantic", "agentic"]:
        try:
            ch = chunk_document(
                path,
                method=m,
                chunk_size=400,
                separators=["\n\n", "\n", ". "],
                n_sentences=4,
                task_description="répondre aux questions",
            )
            print(m, len(ch), "chunks")
        except Exception as e:
            print(m, "error:", e)
