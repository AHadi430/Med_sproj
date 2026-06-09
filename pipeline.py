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
from sentence_transformers import SentenceTransformer, CrossEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ─────────────────────────── CONFIG ───────────────────────────

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
BRAVE_API_KEY   = os.getenv("BRAVE_API_KEY")
GOOGLE_CSE_ID   = os.getenv("GOOGLE_CSE_ID")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")

for name, val in [
    ("GROQ_API_KEY", GROQ_API_KEY),
    ("GEMINI_API_KEY", GEMINI_API_KEY),
    ("GOOGLE_CSE_ID", GOOGLE_CSE_ID),
]:
    if not val:
        raise RuntimeError(f"missing {name} in .env")

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL  = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_EMB  = os.getenv("MODEL_EMB", "gemini-embedding-001")
EMBED_DIM  = 768

TOPK                = 50
TOPN_WEB            = 60
CHUNK_SIZE          = 1200
CHUNK_OVERLAP       = 150
MAX_CHUNKS_PER_PAGE = 12
MAX_NEW_CHUNKS_TO_EMBED_PER_RUN = 20
BATCH_SIZE          = 8
MAX_CONTEXT_TOKENS  = 7000
MIN_RELIABILITY_SCORE = 0.60

# ─────────────────────────── TRUSTED DOMAINS ──────────────────

TRUSTED_MEDICAL_DOMAINS = {
    "pubmed.ncbi.nlm.nih.gov": 1.30,
    "pmc.ncbi.nlm.nih.gov":    1.30,
    "who.int":                 1.30,
    "cdc.gov":                 1.25,
    "nih.gov":                 1.25,
    "thelancet.com":           1.20,
    "nejm.org":                1.20,
    "bmj.com":                 1.20,
    "mayoclinic.org":          1.15,
    "medlineplus.gov":         1.15,
    "nih.org.pk":              1.20,
    "nhsrc.pk":                1.15,
}

HIGH_AUTHORITY_DOMAINS = {
    "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "who.int", "cdc.gov",
    "nih.gov", "medlineplus.gov", "thelancet.com", "nejm.org", "bmj.com",
    "nih.org.pk", "nhsrc.pk",
}

MODERATE_AUTHORITY_SUFFIXES = (
    ".gov", ".edu", ".org",
)

# ─────────────────────────── DISK CACHE ───────────────────────

CACHE_FILE = "./pipeline_cache.json"

def _load_disk_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_disk_cache(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

_disk_cache = _load_disk_cache()

def _cache_key(*parts):
    return hashlib.sha1("||".join(str(p) for p in parts).encode()).hexdigest()

def cached_fetch(url):
    k = _cache_key("fetch", url)
    if k in _disk_cache:
        return _disk_cache[k]
    result = _do_fetch(url)
    if result:
        _disk_cache[k] = result
        _save_disk_cache(_disk_cache)
    return result

def cached_search(engine, query, num):
    k = _cache_key("search", engine, query, num)
    if k in _disk_cache:
        return _disk_cache[k]
    if engine == "google":
        result = _do_google_search(query, num)
    elif engine == "brave":
        result = _do_brave_search(query, num)
    else:
        result = []
    _disk_cache[k] = result
    _save_disk_cache(_disk_cache)
    return result

# ─────────────────────────── GEMINI THROTTLE ──────────────────

MIN_SECONDS_BETWEEN_GEMINI_CALLS = 1.2
MAX_RETRIES    = 6
BACKOFF_BASE   = 1.6
BACKOFF_JITTER = 0.35
_last_gemini_call_ts = 0.0

def _throttle_gemini():
    global _last_gemini_call_ts
    now  = time.time()
    wait = MIN_SECONDS_BETWEEN_GEMINI_CALLS - (now - _last_gemini_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_gemini_call_ts = time.time()

def _is_rate_limit(err):
    s = str(err).lower()
    return ("429" in s) or ("resource_exhausted" in s) or ("rate limit" in s) or ("rpm" in s)

def gemini_call_with_retry(fn):
    for attempt in range(MAX_RETRIES + 1):
        try:
            _throttle_gemini()
            return fn()
        except ClientError as e:
            if (not _is_rate_limit(e)) or attempt == MAX_RETRIES:
                raise
            sleep_s = (BACKOFF_BASE ** attempt) * (1.0 + random.uniform(-BACKOFF_JITTER, BACKOFF_JITTER))
            time.sleep(max(0.5, sleep_s))

def safe_embed_content(model, contents, config):
    return gemini_call_with_retry(
        lambda: gemini_client.models.embed_content(model=model, contents=contents, config=config)
    )

# ─────────────────────────── LLM ──────────────────────────────

class _Resp:
    def __init__(self, text):
        self.text = text

def safe_generate_content(contents, system=None):
    system_msg = system or (
        "answer using only the provided context. "
        "use citations like [1], [2]. "
        "if you cannot answer, say what is missing."
    )
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": contents},
        ],
        temperature=0.2,
    )
    return _Resp((resp.choices[0].message.content or "").strip())

# ─────────────────────────── UTILS ────────────────────────────

def _norm(v):
    v = np.array(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 0 else v.tolist()

def domain_of(url):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def chunk_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    out, step, i = [], max(1, CHUNK_SIZE - CHUNK_OVERLAP), 0
    while i < len(text):
        c = text[i : i + CHUNK_SIZE].strip()
        if c:
            out.append(c)
        i += step
    return out

def _chunk_id(url, chunk):
    h = hashlib.sha1((url + "::" + chunk).encode("utf-8")).hexdigest()
    return f"{url}#sha1:{h}"

def estimate_tokens(text):
    return max(1, len(text) // 4)

def _first_json_object(text):
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None

def _as_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []

def source_summary(hit, index=None):
    meta = hit.get("meta", {})
    item = {
        "url": meta.get("url", ""),
        "title": meta.get("title", ""),
        "domain": meta.get("domain", domain_of(meta.get("url", ""))),
        "similarity": round(float(hit.get("sim", 0.0)), 4),
        "reliability_score": round(float(hit.get("reliability_score", 0.0)), 3),
        "reliability_label": hit.get("reliability_label", "unknown"),
        "reliability_reasons": hit.get("reliability_reasons", []),
    }
    if index is not None:
        item["citation"] = f"[{index}]"
    if "rerank_score" in hit:
        item["rerank_score"] = round(float(hit.get("rerank_score", 0.0)), 4)
    return item

def reliability_label(score):
    if score >= 0.80:
        return "high"
    if score >= MIN_RELIABILITY_SCORE:
        return "moderate"
    return "low"

def score_source_reliability(url, title="", text=""):
    domain = domain_of(url)
    hay = f"{title}\n{text[:5000]}".lower()
    score, reasons = 0.0, []

    if domain in HIGH_AUTHORITY_DOMAINS:
        score += 0.40
        reasons.append("high-authority medical or public-health domain")
    elif any(domain.endswith(s) for s in MODERATE_AUTHORITY_SUFFIXES):
        score += 0.25
        reasons.append("institutional or non-commercial domain")
    else:
        score += 0.12
        reasons.append("general web domain")

    years = [int(y) for y in re.findall(r"\b(20[0-2][0-9]|19[8-9][0-9])\b", hay)]
    if years:
        latest = max(years)
        if latest >= 2021:
            score += 0.20
            reasons.append(f"recent date signal ({latest})")
        elif latest >= 2015:
            score += 0.12
            reasons.append(f"usable but older date signal ({latest})")
        else:
            score += 0.05
            reasons.append(f"old date signal ({latest})")
    else:
        score += 0.04
        reasons.append("no clear publication/data year detected")

    transparency_terms = [
        "methods", "methodology", "sample", "participants", "survey", "registry",
        "confidence interval", "95% ci", "data source", "cohort", "cross-sectional",
        "randomized", "systematic review", "meta-analysis",
    ]
    matched_terms = [t for t in transparency_terms if t in hay]
    if len(matched_terms) >= 3:
        score += 0.20
        reasons.append("methods/data transparency signals present")
    elif matched_terms:
        score += 0.10
        reasons.append("some methods/data transparency signals present")
    else:
        score += 0.03
        reasons.append("limited methods/data transparency signals")

    if any(t in hay for t in ["systematic review", "meta-analysis", "who", "registry", "national survey"]):
        score += 0.15
        reasons.append("consensus or registry-style evidence signal")
    elif any(t in hay for t in ["study", "journal", "doi", "pubmed", "pmc"]):
        score += 0.10
        reasons.append("research-publication evidence signal")
    else:
        score += 0.04
        reasons.append("limited consensus evidence signal")

    score = max(0.0, min(1.0, score))
    return {
        "score": score,
        "label": reliability_label(score),
        "reasons": reasons,
    }

def annotate_hits_reliability(hits):
    for h in hits:
        meta = h.get("meta", {})
        if "reliability_score" in h:
            continue
        scored = score_source_reliability(
            meta.get("url", ""),
            title=meta.get("title", ""),
            text=h.get("text", ""),
        )
        h["reliability_score"] = scored["score"]
        h["reliability_label"] = scored["label"]
        h["reliability_reasons"] = scored["reasons"]
    return hits

def split_sources_by_reliability(hits, threshold=MIN_RELIABILITY_SCORE):
    accepted, rejected = [], []
    seen = set()
    for h in hits:
        url = h.get("meta", {}).get("url", "")
        if url in seen:
            continue
        seen.add(url)
        summary = source_summary(h)
        if h.get("reliability_score", 0.0) >= threshold:
            accepted.append(summary)
        else:
            rejected.append(summary)
    return accepted, rejected

# ─────────────────────────── EMBEDDINGS ───────────────────────

st_model  = SentenceTransformer("all-MiniLM-L6-v2")
reranker  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def gemini_embed_many(texts, task_type):
    cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM)
    res = safe_embed_content(model=MODEL_EMB, contents=texts, config=cfg)
    return [_norm(e.values) for e in res.embeddings], "gemini"

def local_embed_many(texts):
    vecs = st_model.encode(texts, normalize_embeddings=True)
    return [v.astype(np.float32).tolist() for v in vecs], "local"

def embed_many(texts, task_type="RETRIEVAL_DOCUMENT"):
    if os.getenv("FORCE_LOCAL", "0") == "1":
        return local_embed_many(texts)
    try:
        return gemini_embed_many(texts, task_type)
    except ClientError as e:
        if _is_rate_limit(e):
            print("gemini quota hit – falling back to local embeddings")
            return local_embed_many(texts)
        raise

def cosine_sim(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b))

# ─────────────────────────── CHROMADB ─────────────────────────

chroma = chromadb.PersistentClient(path="./chroma_db")
col    = chroma.get_or_create_collection(name="medical_rag_hybrid")

def upsert_chunks(url, title, chunks):
    ids = [_chunk_id(url, c) for c in chunks]
    existing = set()
    try:
        got = col.get(ids=ids, include=[])
        for _id in got.get("ids", []):
            existing.add(_id)
    except Exception:
        pass

    new = [(i, c) for i, c in enumerate(chunks) if ids[i] not in existing]
    if not new:
        return {"stored": 0, "embed_backend": None}

    new = new[:MAX_NEW_CHUNKS_TO_EMBED_PER_RUN]
    new_ids  = [_chunk_id(url, c) for _, c in new]
    new_docs = [c for _, c in new]
    metas    = [{"url": url, "title": title, "domain": domain_of(url)} for _ in new_docs]

    stored, embed_backend = 0, None
    for i in range(0, len(new_docs), BATCH_SIZE):
        bd = new_docs[i : i + BATCH_SIZE]
        bi = new_ids[i : i + BATCH_SIZE]
        bm = metas[i : i + BATCH_SIZE]
        embs, backend = embed_many(bd, task_type="RETRIEVAL_DOCUMENT")
        embed_backend  = backend
        col.upsert(ids=bi, documents=bd, embeddings=embs, metadatas=bm)
        stored += len(bd)

    return {"stored": stored, "embed_backend": embed_backend}

# ─────────────────────────── RETRIEVAL ────────────────────────

def retrieve_topk(query, k=TOPK):
    q_embs, backend = embed_many([query], task_type="RETRIEVAL_QUERY")
    q_emb = q_embs[0]

    res   = col.query(query_embeddings=[q_emb], n_results=k,
                      include=["documents", "metadatas", "embeddings"])
    docs  = res["documents"][0]  if res.get("documents")  else []
    metas = res["metadatas"][0]  if res.get("metadatas")  else []
    embs  = res["embeddings"][0] if res.get("embeddings") else []

    hits = []
    for doc, meta, emb in zip(docs, metas, embs):
        base_sim   = cosine_sim(q_emb, emb)
        trust_mult = TRUSTED_MEDICAL_DOMAINS.get(domain_of(meta.get("url", "")), 1.0)
        hits.append({
            "text": doc,
            "meta": meta,
            "sim":  base_sim * trust_mult,
        })

    hits.sort(key=lambda x: x["sim"], reverse=True)
    hits = annotate_hits_reliability(hits)
    hits = rerank_hits(query, hits)
    return hits, backend

def rerank_hits(query, hits, top_n=20):
    if not hits:
        return hits
    candidates = hits[:top_n]
    pairs  = [(query, h["text"]) for h in candidates]
    scores = reranker.predict(pairs)
    for h, s in zip(candidates, scores):
        h["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates + hits[top_n:]

# ─────────────────────────── SEARCH ───────────────────────────

def _do_google_search(query, num=10):
    if not GOOGLE_API_KEY:
        return []
    r = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={"q": query, "key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "num": min(num, 10)},
        timeout=20,
    )
    if r.status_code == 403:
        return []
    r.raise_for_status()
    return [
        {"title": it.get("title",""), "url": it.get("link",""),
         "snippet": it.get("snippet",""), "domain": domain_of(it.get("link",""))}
        for it in r.json().get("items", [])
    ]

def _do_brave_search(query, num=10):
    if not BRAVE_API_KEY:
        return []
    r = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": min(num, 20)},
        headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
        timeout=20,
    )
    r.raise_for_status()
    return [
        {"title": it.get("title",""), "url": it.get("url",""),
         "snippet": it.get("description",""), "domain": domain_of(it.get("url",""))}
        for it in r.json().get("web", {}).get("results", [])
    ]

def google_search(query, num=10):
    return cached_search("google", query, num)

def brave_search(query, num=10):
    return cached_search("brave", query, num)

# ─────────────────────────── FETCH ────────────────────────────

def _do_fetch(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_tables=True)
        if not text or len(text) < 300:
            return None
        return text
    except Exception:
        return None

def fetch_and_extract(url):
    return cached_fetch(url)

# ─────────────────────────── QUERY HELPERS ────────────────────

def build_search_queries(q):
    prompt = (
        f'generate 3-5 diverse search queries for retrieving medical information about:\n"{q}"\n\n'
        "rules:\n"
        "- include medical terms, synonyms, and related concepts\n"
        "- add terms like statistics, epidemiology, prevalence\n"
        "- keep queries concise (5-10 words)\n"
        "- one query per line, no numbering\n\n"
        "queries:"
    )
    try:
        resp = safe_generate_content(prompt)
        queries = [l.strip("-• 0123456789.) ").strip() for l in resp.text.splitlines()
                   if len(l.strip()) > 10]
        return [q] + queries[:4]
    except Exception:
        return [q]

def expand_query_regionally(query):
    prompt = (
        "you are helping retrieve publicly available medical statistics.\n\n"
        "expand the following query by progressively broadening only its "
        "geographic scope (city -> province/state -> country -> region -> global).\n\n"
        "rules:\n"
        "- do not change disease, metric, or population\n"
        "- do not invent numbers\n"
        "- do not include explanations\n"
        "- each line must be a valid search query\n\n"
        f"original query:\n{query}\n\n"
        "expanded queries (one per line):"
    )
    try:
        resp = safe_generate_content(prompt)
        lines = (resp.text or "").splitlines()
        expanded = [l.strip("-• 0123456789.) ").strip() for l in lines if len(l.strip()) > 10]
        out = [query]
        for q in expanded:
            if q.lower() != query.lower():
                out.append(q)
        return out[:4]
    except Exception:
        return [query]

def decompose_query(query, query_type="general"):
    prompt = (
        "decompose this medical/statistical query into strict JSON only.\n\n"
        "schema:\n"
        "{\n"
        '  "population": {"location": null, "age": null, "gender": null, "group": null},\n'
        '  "condition": null,\n'
        '  "metric": null,\n'
        '  "time_period": null,\n'
        '  "comparators": [],\n'
        '  "disease_definition": null,\n'
        '  "requires_sql_rag": false,\n'
        '  "requires_web_rag": true,\n'
        '  "needs_clarification": false,\n'
        '  "missing_inputs": [],\n'
        '  "assumptions": []\n'
        "}\n\n"
        "rules:\n"
        "- use null when a field is not stated\n"
        "- requires_sql_rag is true for my practice, my patients, my community, local database, Zanjabee, or patient dataset\n"
        "- needs_clarification is true when a required clinical definition or population is ambiguous\n"
        "- do not invent facts or numbers\n\n"
        f"query type: {query_type}\n"
        f"query: {query}\n"
    )
    fallback = {
        "population": {"location": None, "age": None, "gender": None, "group": None},
        "condition": None,
        "metric": None,
        "time_period": None,
        "comparators": [],
        "disease_definition": None,
        "requires_sql_rag": bool(re.search(r"\b(my practice|my patients|my community|zanjabee|local db|local database)\b", query.lower())),
        "requires_web_rag": True,
        "needs_clarification": False,
        "missing_inputs": [],
        "assumptions": [],
    }
    try:
        resp = safe_generate_content(
            prompt,
            system="you extract structured query fields. respond with valid JSON only.",
        )
        parsed = _first_json_object(resp.text)
        if not isinstance(parsed, dict):
            return fallback
        out = fallback.copy()
        out.update(parsed)
        if not isinstance(out.get("population"), dict):
            out["population"] = fallback["population"]
        out["comparators"] = _as_list(out.get("comparators"))
        out["missing_inputs"] = _as_list(out.get("missing_inputs"))
        out["assumptions"] = _as_list(out.get("assumptions"))
        out["requires_sql_rag"] = bool(out.get("requires_sql_rag"))
        out["requires_web_rag"] = bool(out.get("requires_web_rag", True))
        out["needs_clarification"] = bool(out.get("needs_clarification"))
        return out
    except Exception:
        return fallback

def _join_query_parts(*parts):
    return re.sub(r"\s+", " ", " ".join(str(p) for p in parts if p)).strip()

def _base_query_from_plan(plan, fallback_query):
    pop = plan.get("population", {}) if isinstance(plan.get("population"), dict) else {}
    parts = [
        plan.get("metric"),
        plan.get("condition"),
        pop.get("group"),
        pop.get("gender"),
        pop.get("age"),
        pop.get("location"),
        plan.get("time_period"),
    ]
    q = _join_query_parts(*parts)
    return q if len(q.split()) >= 3 else fallback_query

def _replace_location(query, old, new):
    if not old or not new or old == new:
        return query
    return re.sub(re.escape(old), new, query, flags=re.IGNORECASE)

def _location_expansions(location):
    if not location:
        return []
    loc = str(location).strip()
    lower = loc.lower()
    generic = {
        "dha phase 5": ["DHA Lahore", "Lahore", "Punjab", "Pakistan"],
        "dha": ["Lahore", "Punjab", "Pakistan"],
        "lahore": ["Punjab", "Pakistan", "South Asia"],
        "islamabad": ["Pakistan", "South Asia"],
        "punjab": ["Pakistan", "South Asia"],
        "kpk": ["Khyber Pakhtunkhwa", "Pakistan", "South Asia"],
        "khyber pakhtunkhwa": ["Pakistan", "South Asia"],
        "karachi": ["Sindh", "Pakistan", "South Asia"],
        "sindh": ["Pakistan", "South Asia"],
        "balochistan": ["Pakistan", "South Asia"],
        "pakistan": ["South Asia", "global"],
    }
    return generic.get(lower, ["Pakistan", "South Asia", "global"])

def build_expansion_plan(query, plan, query_type="general"):
    base = _base_query_from_plan(plan, query)
    pop = plan.get("population", {}) if isinstance(plan.get("population"), dict) else {}
    location = pop.get("location")
    time_period = plan.get("time_period")
    condition = plan.get("condition")
    metric = plan.get("metric")

    levels = [{
        "level": 0,
        "axis": "exact",
        "query": base,
        "description": "exact requested criteria",
        "changes": {},
    }]

    seen = {base.lower()}
    for loc in _location_expansions(location):
        q = _replace_location(base, location, loc)
        if q and q.lower() not in seen:
            seen.add(q.lower())
            levels.append({
                "level": len(levels),
                "axis": "population",
                "query": q,
                "description": f"broadened location from {location} to {loc}",
                "changes": {"location_from": location, "location_to": loc},
            })
        if len(levels) >= 4:
            break

    if time_period and len(levels) < 5:
        q = re.sub(re.escape(str(time_period)), "", base, flags=re.IGNORECASE).strip()
        q = re.sub(r"\s+", " ", q)
        if q and q.lower() not in seen:
            seen.add(q.lower())
            levels.append({
                "level": len(levels),
                "axis": "time",
                "query": q,
                "description": f"removed time constraint: {time_period}",
                "changes": {"time_period_removed": time_period},
            })

    if query_type != "regional" and len(levels) < 4:
        for q in build_search_queries(query)[1:]:
            if q.lower() not in seen:
                seen.add(q.lower())
                levels.append({
                    "level": len(levels),
                    "axis": "rephrase",
                    "query": q,
                    "description": "LLM-generated retrieval rephrase",
                    "changes": {},
                })
            if len(levels) >= 4:
                break

    if condition and metric and len(levels) < 5:
        q = _join_query_parts(condition, metric, "epidemiology statistics")
        if q.lower() not in seen:
            levels.append({
                "level": len(levels),
                "axis": "disease",
                "query": q,
                "description": "broadened to disease and metric without population filters",
                "changes": {"removed_population_filters": True},
            })

    return levels[:5]

def classify_query(query):
    prompt = (
        'classify this medical query into exactly one category:\n'
        '- "definition": asking what something is\n'
        '- "statistics": asking for numbers, rates, prevalence\n'
        '- "treatment": asking about protocols, medications\n'
        '- "regional": asking for location-specific data\n\n'
        f'query: "{query}"\n\n'
        'respond with only the category word, nothing else.'
    )
    try:
        resp = safe_generate_content(prompt)
        cat  = resp.text.strip().lower().strip('"')
        return cat if cat in {"definition", "statistics", "treatment", "regional"} else "general"
    except Exception:
        return "general"

# ─────────────────────────── PRIVACY GATE ─────────────────────

def should_skip_web(query):
    q = query.lower()
    patient_patterns = [
        r"\bpatient\s+id\b", r"\bmrn\b", r"\bmedical\s+record\b",
        r"\bcase\s+file\b",   r"\bclinical\s+note\b", r"\bdischarge\s+summary\b",
        r"\bimaging\s+(scan|report)\b", r"\bmri\b", r"\bct\s+scan\b",
        r"\bx[-\s]?ray\b",    r"\becg\b", r"\behr\b",
    ]
    internal_patterns = [
        r"\binternal\b", r"\bconfidential\b", r"\bprivate\b",
        r"\bunpublished\b",   r"\bin-house\b",
    ]
    for p in patient_patterns + internal_patterns:
        if re.search(p, q):
            return True
    if re.search(r"\b(patient|person|individual)\s+[A-Z][a-z]+\b", query):
        return True
    return False

# ─────────────────────────── RETRIEVAL QUALITY ────────────────

def retrieval_is_promising(hits, min_hits=3, min_sim=0.35, min_diversity=2):
    if len(hits) < min_hits:
        return False
    avg_sim = sum(h["sim"] for h in hits[:min_hits]) / min_hits
    if avg_sim < min_sim:
        return False
    domains = {h["meta"].get("domain", "") for h in hits[:5]}
    return len(domains) >= min_diversity

def llm_response_indicates_deferral(ans, query_type="general"):
    ans_lower = ans.lower()
    deferral_phrases = [
        "not available", "no data for", "would require",
        "insufficient data", "not reported", "only available at",
        "no published statistics", "unknown at this level",
        "i cannot answer", "context does not contain",
        "not mentioned in the provided context",
        "not mentioned in the context",
        "does not contain information",
        "does not provide information",
        "no information about",
        "none of them mention",
        "not possible to find",
        "what is missing",
        "missing is data",
        "missing is information",
    ]
    if any(p in ans_lower for p in deferral_phrases):
        return True
    ans_without_citations = re.sub(r"\[\d+\]", "", ans_lower)
    if query_type in {"statistics", "regional"} and not re.search(r"\d", ans_without_citations):
        return True
    return False

# ─────────────────────────── CONTEXT PACKING ──────────────────

def pack_hits_by_token_budget(hits, max_tokens=MAX_CONTEXT_TOKENS):
    packed, used = [], 0
    for h in hits:
        t = estimate_tokens(h["text"])
        if used + t > max_tokens:
            break
        packed.append(h)
        used += t
    return packed, used

def llm_answer_or_context(query, hits, query_plan=None, expansion=None):
    packed_hits, _ = pack_hits_by_token_budget(hits)
    if not packed_hits:
        return "no retrieved context available.", "no_context"

    blocks = []
    for i, h in enumerate(packed_hits, 1):
        url = h["meta"].get("url", "")
        reliability = h.get("reliability_score", 0.0)
        label = h.get("reliability_label", "unknown")
        blocks.append(f"[{i}] {url}\nreliability: {label} ({reliability:.2f})\n{h['text']}")
    context = "\n\n".join(blocks)
    plan_text = json.dumps(query_plan or {}, ensure_ascii=False, indent=2)
    expansion_text = json.dumps(expansion or {}, ensure_ascii=False, indent=2)

    prompt = (
        f"question:\n{query}\n\n"
        f"structured query plan:\n{plan_text}\n\n"
        f"expansion used:\n{expansion_text}\n\n"
        f"context:\n{context}\n\n"
        "answer using only the context above. use citations like [1], [2]. "
        "include a short transparency note covering exact vs expanded criteria, "
        "assumptions, what was found, and what was not found. "
        "if you cannot answer, say exactly what is missing."
    )
    try:
        resp = safe_generate_content(prompt)
        text = (resp.text or "").strip()
        return (text or "empty llm response.", "empty_llm" if not text else "llm")
    except ClientError as e:
        if _is_rate_limit(e):
            return "llm quota exhausted. context:\n\n" + context, "no_llm"
        raise

# ─────────────────────────── WEB INGEST ───────────────────────

def web_ingest(query, use_brave=True, debug=True):
    squeries = build_search_queries(query)
    if debug:
        print("\nsearch queries:")
        for q in squeries:
            print(" -", q)

    urls, seen = [], set()
    for sq in squeries:
        try:
            results = google_search(sq, num=TOPN_WEB)
        except Exception as e:
            if debug:
                print("search error:", e)
            results = []
        for r in results:
            u = r.get("url", "")
            if u and u not in seen:
                seen.add(u)
                urls.append(r)

    if debug:
        print(f"\nfound {len(urls)} urls")

    extracted_ok = total_stored = 0
    last_backend = None

    def _fetch_one(r):
        return r, fetch_and_extract(r["url"])

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(_fetch_one, r) for r in urls[:TOPN_WEB]]
        for fut in as_completed(futures):
            r, text = fut.result()
            if not text:
                continue
            extracted_ok += 1
            chunks = chunk_text(text)[:MAX_CHUNKS_PER_PAGE]
            stats  = upsert_chunks(r["url"], r.get("title", ""), chunks)
            total_stored += stats["stored"]
            last_backend  = stats["embed_backend"]

    if debug:
        print(f"extracted: {extracted_ok}  stored new chunks: {total_stored}  backend: {last_backend}")

    return {"search_queries": squeries, "results": urls[:TOPN_WEB], "chunks_added": total_stored}

# ─────────────────────────── UNIFIED PIPELINE ─────────────────

def _output_payload(mode, answer, hits, retrieval_backend, llm_mode, query_used,
                    expanded, query_type, query_plan, expansion_plan,
                    expansions_used=None, assumptions=None, missing=None,
                    reason=None, expansion_level=None, source_pool=None):
    hits = annotate_hits_reliability(hits or [])
    source_pool = annotate_hits_reliability(source_pool or hits)
    accepted_sources, rejected_sources = split_sources_by_reliability(source_pool)
    payload = {
        "mode": mode,
        "answer": answer,
        "hits": hits,
        "retrieval_backend": retrieval_backend,
        "llm_mode": llm_mode,
        "query_used": query_used,
        "expanded": expanded,
        "query_type": query_type,
        "query_plan": query_plan,
        "expansion_plan": expansion_plan,
        "expansions_used": expansions_used or [],
        "accepted_sources": accepted_sources,
        "rejected_sources": rejected_sources,
        "assumptions": assumptions or [],
        "missing": missing or [],
    }
    if reason:
        payload["reason"] = reason
    if expansion_level is not None:
        payload["expansion_level"] = expansion_level
    return payload

def run_pipeline(query, debug=True):
    query      = query.strip()
    query_type = classify_query(query)
    query_plan = decompose_query(query, query_type=query_type)
    expansion_plan = build_expansion_plan(query, query_plan, query_type=query_type)
    planned_assumptions = _as_list(query_plan.get("assumptions"))
    planned_missing = _as_list(query_plan.get("missing_inputs"))

    if debug:
        print("=" * 70)
        print(f"query : {query}")
        print(f"type  : {query_type}")
        print(f"plan  : {json.dumps(query_plan, ensure_ascii=False)}")
        print("=" * 70)

    min_sim_threshold = 0.30 if query_type == "definition" else 0.35

    # step 1: local vector search
    hits, emb_backend = retrieve_topk(query)
    accepted_hits = [h for h in hits if h.get("reliability_score", 0.0) >= MIN_RELIABILITY_SCORE]
    answer_hits = accepted_hits if len(accepted_hits) >= 2 else hits
    ans, llm_mode = "", "no_llm"

    if retrieval_is_promising(answer_hits, min_sim=min_sim_threshold):
        ans, llm_mode = llm_answer_or_context(
            query,
            answer_hits,
            query_plan=query_plan,
            expansion=expansion_plan[0] if expansion_plan else None,
        )

    weak = (
        not answer_hits or
        llm_response_indicates_deferral(ans, query_type) or
        len(ans) < 120
    )

    if not weak:
        return _output_payload(
            "local", ans, answer_hits, emb_backend, llm_mode, query, False,
            query_type, query_plan, expansion_plan,
            assumptions=planned_assumptions,
            missing=planned_missing,
            source_pool=hits,
        )

    # step 2: privacy gate
    if should_skip_web(query):
        return _output_payload(
            "blocked", "insufficient public data.", [], None, "blocked", query, False,
            query_type, query_plan, expansion_plan,
            assumptions=planned_assumptions,
            missing=planned_missing,
            reason="query involves private or non-public information",
        )

    if debug:
        print("\nexpansions:")
        for item in expansion_plan:
            print(f"  [{item['level']}] {item['axis']}: {item['query']}")

    # step 4: web ingest + re-retrieve
    for expansion in expansion_plan:
        level = expansion["level"]
        q_exp = expansion["query"]
        if debug:
            print(f"\n--- expansion level {level} ({expansion['axis']}): {q_exp} ---")

        web_ingest(q_exp, debug=debug)
        hits_exp, be_exp = retrieve_topk(q_exp)
        accepted_exp = [h for h in hits_exp if h.get("reliability_score", 0.0) >= MIN_RELIABILITY_SCORE]
        answer_exp_hits = accepted_exp if len(accepted_exp) >= 2 else hits_exp

        if not retrieval_is_promising(answer_exp_hits, min_sim=min_sim_threshold):
            continue

        ans_exp, llm_exp = llm_answer_or_context(
            q_exp,
            answer_exp_hits,
            query_plan=query_plan,
            expansion=expansion,
        )
        if not llm_response_indicates_deferral(ans_exp, query_type) and len(ans_exp) >= 120:
            expansions_used = []
            if level > 0:
                expansions_used = [e for e in expansion_plan if 0 < e["level"] <= level]
            assumptions = planned_assumptions[:]
            if level > 0:
                assumptions.append(expansion["description"])
            if len(accepted_exp) < len(hits_exp):
                assumptions.append(
                    f"{len(hits_exp) - len(accepted_exp)} lower-reliability sources were excluded or caveated"
                )
            return _output_payload(
                "web", ans_exp, answer_exp_hits, be_exp, llm_exp, q_exp, level > 0,
                query_type, query_plan, expansion_plan,
                expansions_used=expansions_used,
                assumptions=assumptions,
                missing=planned_missing,
                expansion_level=level,
                source_pool=hits_exp,
            )

    missing = planned_missing[:]
    missing.append("reliable published data matching the requested criteria")
    return _output_payload(
        "failed", "no reliable published data found.", [], None, "no_data", query, True,
        query_type, query_plan, expansion_plan,
        expansions_used=[e for e in expansion_plan if e["level"] > 0],
        assumptions=planned_assumptions,
        missing=missing,
    )
