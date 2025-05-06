from mistralai import Mistral
import requests
import numpy as np
import faiss
import os
import time
import json
from pathlib import Path

from Chunking import *

# Initialize API client
api_key = "3nIsGfqZ6bdMteCc4qsBHjyjcGHZUSrm"
client = Mistral(api_key=api_key)

# Constants for caching
CACHE_DIR = Path("chunk_cache")
EMBED_CACHE_DIR = CACHE_DIR / "embeddings"
EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Fetch document path
doc_path = Path(
    "Syllabus_Ing_2024_2025.pdf"
)

# ================= Pipeline parameters =================
method = (
    "semantic"  # choose method: fixed, recursive, page, sentence, semantic, agentic
)
kwargs = {
    "chunk_size": 400,
    "separators": ["\n\n", "\n", ". "],
    "n_sentences": 4,
    "task_description": "répondre aux questions de l'étudiant de Centrale",
}


# ================= Helper: caching chunks =================
def load_cached_chunks(doc_path: Path, method: str) -> list | None:
    cache_file = CACHE_DIR / f"{doc_path.stem}_{method}_chunks.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    return None


def save_cached_chunks(doc_path: Path, method: str, chunks: list):
    cache_file = CACHE_DIR / f"{doc_path.stem}_{method}_chunks.json"
    cache_file.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ================= Helper: caching embeddings =================
def load_cached_embeddings(doc_path: Path, method: str) -> np.ndarray | None:
    cache_file = EMBED_CACHE_DIR / f"{doc_path.stem}_{method}_embeddings.npy"
    if cache_file.exists():
        return np.load(cache_file)
    return None


def save_cached_embeddings(doc_path: Path, method: str, embeddings: np.ndarray):
    cache_file = EMBED_CACHE_DIR / f"{doc_path.stem}_{method}_embeddings.npy"
    np.save(cache_file, embeddings)


# ================= Load or compute chunks =================
cached_chunks = load_cached_chunks(doc_path, method)
if cached_chunks is not None:
    chunks = cached_chunks
    print(f"Loaded {len(chunks)} cached chunks for method '{method}'")
else:
    chunks = chunk_document(doc_path, method=method, **kwargs)
    save_cached_chunks(doc_path, method, chunks)
    print(f"Computed and cached {len(chunks)} chunks for method '{method}'")

# ================= Embedding caching =================
cached_embeds = load_cached_embeddings(doc_path, method)
if cached_embeds is not None and cached_embeds.shape[0] == len(chunks):
    text_embeddings = cached_embeds
    print(
        f"Loaded cached embeddings for method '{method}' ({text_embeddings.shape[0]} vectors)"
    )


def get_text_embedding(inputs):
    time.sleep(2)
    response = client.embeddings.create(model="mistral-embed", inputs=inputs)
    return [emb.embedding for emb in response.data]


# Adaptive batching
def estimate_tokens(text: str) -> int:
    return len(text.split())


def make_batches_by_token_limit(chunks, max_tokens=1000):
    batches, batch, tokens = [], [], 0
    for c in chunks:
        t = estimate_tokens(c)
        if t > max_tokens:
            if batch:
                batches.append(batch)
                batch, tokens = [], 0
            batches.append([c])
            continue
        if tokens + t <= max_tokens:
            batch.append(c)
            tokens += t
        else:
            batches.append(batch)
            batch, tokens = [c], t
    if batch:
        batches.append(batch)
    return batches


# Only compute embeddings if not cached
def compute_embeddings(chunks):
    batches = make_batches_by_token_limit(chunks, max_tokens=1000)
    embeds = []
    for idx, batch in enumerate(batches, 1):
        total = sum(estimate_tokens(c) for c in batch)
        print(f"Embedding batch {idx}/{len(batches)} (~{total} tokens)")
        embeds.extend(get_text_embedding(batch))
    arr = np.array(embeds)
    return arr


if cached_embeds is None or cached_embeds.shape[0] != len(chunks):
    text_embeddings = compute_embeddings(chunks)
    save_cached_embeddings(doc_path, method, text_embeddings)
    print(
        f"Computed and cached {text_embeddings.shape[0]} embeddings for method '{method}'"
    )

# ================= FAISS index build =================
d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)


# ================= RAG function =================
def run_mistral(question, model="mistral-large-latest"):
    # Query processing
    time.sleep(2)
    question_embeddings = np.array(get_text_embedding([question]))  # Batch single query

    D, I = index.search(question_embeddings, k=4)  # Retrieve top 2 relevant chunks
    retrieved_chunk = [chunks[i] for i in I[0]]

    messages = [
        {
            "role": "user",
            "content": f"""
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
""",
        }
    ]
    time.sleep(2)
    chat_response = client.chat.complete(model=model, messages=messages)
    return chat_response.choices[0].message.content


# ================= Example =================
if __name__ == "__main__":
    print(
        run_mistral(
            "Quels sont les objectifs d'apprentissage pour l'UE économie-gestion ?"
        )
    )

# if __name__ == "__main__":
#     question = "Quels sont les objectifs d'apprentissage pour l'UE économie-gestion ?"
#     methods = ["fixed", "recursive", "page", "sentence", "semantic"]
#     results = {}
#     for m in methods:
#         print(f"\n--- Méthode: {m} ---")
#         # load or compute chunks for this method
#         cached_chunks = load_cached_chunks(doc_path, m)
#         if cached_chunks is not None:
#             chunks = cached_chunks
#         else:
#             chunks = chunk_document(doc_path, method=m, **kwargs)
#             save_cached_chunks(doc_path, m, chunks)
#         # load or compute embeddings
#         cached_embeds = load_cached_embeddings(doc_path, m)
#         if cached_embeds is not None and cached_embeds.shape[0] == len(chunks):
#             text_embeddings = cached_embeds
#         else:
#             text_embeddings = compute_embeddings(chunks)
#             save_cached_embeddings(doc_path, m, chunks)
#         # build index
#         # d = text_embeddings.shape[1]
#         # temp_index = faiss.IndexFlatL2(d)
#         # temp_index.add(text_embeddings)
#         # run query
#         # temporarily override global index and chunks
#         # old_index, old_chunks = index, chunks
#         # index = temp_index
#         answer = run_mistral(question)
#         results[m] = answer
#         print(answer)
#         # restore
#         # index, chunks = old_index, old_chunks
#     # Summary
#     print("\n=== Résumé des réponses ===")
#     for m, ans in results.items():
#         print(f"\n{m}: {ans}")
