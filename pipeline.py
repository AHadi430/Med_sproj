import os, re, time, random, hashlib, json, requests
from urllib.parse import urlparse
from dotenv import load_dotenv
import trafilatura
import numpy as np
import chromadb
from groq import Groq
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ───────────────────────── CONFIG ─────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MODEL_EMB  = os.getenv("MODEL_EMB", "gemini-embedding-001")
EMBED_DIM   = 768

TOPK = 40
TOPN_WEB = 40
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 120
BATCH_SIZE = 6
MAX_CONTEXT_TOKENS = 6500
MIN_RELIABILITY_SCORE = 0.60

# ───────────────────────── LAZY MODELS (CRITICAL FIX) ─────────────────────────

_st_model = None
_reranker = None

def get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model

def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

# ───────────────────────── CACHE ─────────────────────────

CACHE_FILE = "./pipeline_cache.json"

def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            return json.load(open(CACHE_FILE, "r", encoding="utf-8"))
        except:
            return {}
    return {}

_disk_cache = _load_cache()

def _save_cache():
    try:
        json.dump(_disk_cache, open(CACHE_FILE, "w", encoding="utf-8"))
    except:
        pass

def _ck(*p):
    return hashlib.sha1("||".join(map(str, p)).encode()).hexdigest()

# ───────────────────────── EMBEDDINGS ─────────────────────────

def _norm(v):
    v = np.array(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / n).tolist() if n else v.tolist()

def gemini_embed(texts, task):
    cfg = types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIM)
    res = gemini_client.models.embed_content(
        model=MODEL_EMB,
        contents=texts,
        config=cfg
    )
    return [_norm(e.values) for e in res.embeddings]

def local_embed(texts):
    model = get_st_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return [v.astype(np.float32).tolist() for v in vecs]

def embed(texts, task="RETRIEVAL_DOCUMENT"):
    try:
        return gemini_embed(texts, task)
    except Exception:
        return local_embed(texts)

# ───────────────────────── CHROMA ─────────────────────────

chroma = chromadb.PersistentClient(path="./chroma_db")
col = chroma.get_or_create_collection("medical_rag_hybrid")

def chunk(text):
    text = re.sub(r"\s+", " ", text)
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [text[i:i+CHUNK_SIZE] for i in range(0, len(text), step)]

def cid(url, c):
    return f"{url}#{hashlib.sha1(c.encode()).hexdigest()}"

def upsert(url, title, chunks):
    if not chunks:
        return 0

    chunks = chunks[:10]  # memory cap

    ids = [cid(url, c) for c in chunks]
    metas = [{"url": url, "title": title} for _ in chunks]

    embs = embed(chunks)

    col.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embs,
        metadatas=metas
    )
    return len(chunks)

# ───────────────────────── SEARCH + FETCH ─────────────────────────

def search_google(q, n=10):
    if not GOOGLE_API_KEY:
        return []
    r = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={"q": q, "key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "num": min(n, 10)},
        timeout=15
    )
    if r.status_code != 200:
        return []
    return [
        {"url": i.get("link"), "title": i.get("title", "")}
        for i in r.json().get("items", [])
    ]

def fetch(url):
    try:
        d = trafilatura.fetch_url(url)
        if not d:
            return None
        return trafilatura.extract(d)
    except:
        return None

# ───────────────────────── RETRIEVAL ─────────────────────────

def retrieve(query):
    q_emb = embed([query], "RETRIEVAL_QUERY")[0]

    res = col.query(
        query_embeddings=[q_emb],
        n_results=TOPK,
        include=["documents", "metadatas", "embeddings"]
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    embs = res.get("embeddings", [[]])[0]

    hits = []
    for d, m, e in zip(docs, metas, embs):
        sim = float(np.dot(q_emb, e))
        hits.append({"text": d, "meta": m, "sim": sim})

    return sorted(hits, key=lambda x: x["sim"], reverse=True)

def rerank(query, hits):
    model = get_reranker()
    pairs = [(query, h["text"]) for h in hits[:15]]
    scores = model.predict(pairs)

    for h, s in zip(hits[:15], scores):
        h["score"] = float(s)

    return sorted(hits, key=lambda x: x.get("score", 0), reverse=True)

# ───────────────────────── INGEST (MEMORY SAFE) ─────────────────────────

def ingest(query):
    urls = search_google(query, TOPN_WEB)

    def worker(u):
        text = fetch(u["url"])
        if not text:
            return 0
        chunks = chunk(text)[:8]
        return upsert(u["url"], u.get("title", ""), chunks)

    stored = 0

    with ThreadPoolExecutor(max_workers=3) as ex:
        for r in ex.map(worker, urls[:TOPN_WEB]):
            stored += r

    return stored

# ───────────────────────── LLM ─────────────────────────

def llm(prompt):
    return groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "use only context, cite [1], [2]"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    ).choices[0].message.content

# ───────────────────────── PIPELINE ─────────────────────────

def run_pipeline(query):
    hits = rerank(query, retrieve(query))

    top = hits[:8]

    if len(top) < 2:
        ingest(query)
        hits = rerank(query, retrieve(query))
        top = hits[:8]

    context = "\n\n".join(
        f"[{i+1}] {h['meta'].get('url','')}\n{h['text'][:1200]}"
        for i, h in enumerate(top)
    )

    prompt = f"""
question: {query}

context:
{context}

answer using only context with citations [1], [2]
"""

    return llm(prompt)

# ───────────────────────── CLEANUP HELPERS ─────────────────────────

def clear_memory():
    global _st_model, _reranker
    _st_model = None
    _reranker = None