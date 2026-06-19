import os, re, json, time, hashlib
import numpy as np
from urllib.parse import urlparse

import chromadb
from dotenv import load_dotenv
from groq import Groq
import trafilatura

load_dotenv()

# ───────────────────────── CONFIG ─────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
TOPK = 40
MAX_CONTEXT = 6000

# ───────────────────────── LAZY MODELS ─────────────────────────

_st_model = None
_reranker = None

def get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(EMBED_MODEL)
    return _st_model


def get_reranker():
    """
    OPTIONAL: can be disabled to save RAM completely.
    """
    global _reranker
    if os.getenv("DISABLE_RERANKER", "1") == "1":
        return None

    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

# ───────────────────────── CHROMA (LAZY) ─────────────────────────

_col = None

def get_col():
    global _col
    if _col is None:
        client = chromadb.PersistentClient(path="./chroma_db")
        _col = client.get_or_create_collection("medical_rag")
    return _col

# ───────────────────────── EMBEDDINGS ─────────────────────────

def embed(texts):
    model = get_st_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return [v.astype(np.float32).tolist() for v in vecs]

# ───────────────────────── UTILS ─────────────────────────

def domain(url):
    return urlparse(url).netloc.lower()

def chunk_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [text[i:i+CHUNK_SIZE] for i in range(0, len(text), step)]

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b))

# ───────────────────────── FETCH ─────────────────────────

def fetch(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        return trafilatura.extract(downloaded)
    except:
        return None

# ───────────────────────── INGEST ─────────────────────────

def upsert(url, title, chunks):
    col = get_col()
    ids = [hashlib.sha1((url+c).encode()).hexdigest() for c in chunks]

    col.upsert(
        ids=ids,
        documents=chunks,
        metadatas=[{"url": url, "title": title} for _ in chunks],
        embeddings=embed(chunks)
    )

# ───────────────────────── RETRIEVE ─────────────────────────

def retrieve(query, k=TOPK):
    col = get_col()

    qvec = embed([query])[0]

    res = col.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "embeddings"]
    )

    hits = []
    for doc, meta, emb in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["embeddings"][0]
    ):
        hits.append({
            "text": doc,
            "meta": meta,
            "score": cosine(qvec, emb)
        })

    return sorted(hits, key=lambda x: x["score"], reverse=True)

# ───────────────────────── RERANK (SAFE) ─────────────────────────

def rerank(query, hits):
    reranker = get_reranker()

    if reranker is None:
        return hits  # fallback: no torch

    pairs = [(query, h["text"]) for h in hits[:20]]
    scores = reranker.predict(pairs)

    for h, s in zip(hits[:20], scores):
        h["rerank"] = float(s)

    return sorted(hits, key=lambda x: x.get("rerank", x["score"]), reverse=True)

# ───────────────────────── LLM ─────────────────────────

def answer(query, hits):
    context = "\n\n".join(
        f"[{i+1}] {h['text']}"
        for i, h in enumerate(hits[:8])
    )

    prompt = f"""
Question: {query}

Context:
{context}

Answer using only context. Use citations like [1], [2].
"""

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return resp.choices[0].message.content

# ───────────────────────── PIPELINE ─────────────────────────

def run(query):
    # 1. retrieve
    hits = retrieve(query)

    # 2. rerank (optional)
    hits = rerank(query, hits)

    # 3. answer
    return {
        "answer": answer(query, hits),
        "sources": [h["meta"] for h in hits[:5]]
    }