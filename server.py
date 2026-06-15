import json
import os
import traceback
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
HOST = "127.0.0.1"
PORT = int(os.getenv("PORT", "8765"))


def _safe_hit(hit):
    meta = hit.get("meta", {}) if isinstance(hit, dict) else {}
    text = hit.get("text", "") if isinstance(hit, dict) else ""
    return {
        "text": text[:900],
        "meta": {
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "domain": meta.get("domain", ""),
        },
        "sim": float(hit.get("sim", 0.0)) if isinstance(hit, dict) else 0.0,
        "rerank_score": float(hit.get("rerank_score", 0.0)) if isinstance(hit, dict) and "rerank_score" in hit else None,
        "reliability_score": float(hit.get("reliability_score", 0.0)) if isinstance(hit, dict) else 0.0,
        "reliability_label": hit.get("reliability_label", "unknown") if isinstance(hit, dict) else "unknown",
        "reliability_reasons": hit.get("reliability_reasons", []) if isinstance(hit, dict) else [],
    }


def _safe_result(out):
    hits = out.get("hits", []) if isinstance(out, dict) else []
    return {
        "mode": out.get("mode", "unknown"),
        "answer": out.get("answer", ""),
        "query_type": out.get("query_type", "unknown"),
        "query_used": out.get("query_used", ""),
        "expanded": out.get("expanded", False),
        "expansion_level": out.get("expansion_level"),
        "retrieval_backend": out.get("retrieval_backend"),
        "llm_mode": out.get("llm_mode"),
        "query_plan": out.get("query_plan", {}),
        "expansion_plan": out.get("expansion_plan", []),
        "expansions_used": out.get("expansions_used", []),
        "accepted_sources": out.get("accepted_sources", []),
        "rejected_sources": out.get("rejected_sources", []),
        "assumptions": out.get("assumptions", []),
        "missing": out.get("missing", []),
        "reason": out.get("reason"),
        "hits": [_safe_hit(h) for h in hits[:8]],
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")

        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json(200, {"ok": True})
            return
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self._json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            query = (payload.get("query") or "").strip()
            debug = bool(payload.get("debug", False))
            if not query:
                self._json(400, {"error": "query is required"})
                return

            from pipeline import run_pipeline

            out = run_pipeline(query, debug=debug)
            self._json(200, {"ok": True, "result": _safe_result(out)})
        except Exception as exc:
            self._json(500, {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=4),
            })
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main():
    STATIC_DIR.mkdir(exist_ok=True)
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Serving medical RAG chat at http://{HOST}:{PORT}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
