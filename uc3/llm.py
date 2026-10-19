"""Model backends. Everything goes through `backend.chat(system, user) -> str` so models are swappable."""
import json
import re
import requests

from .retrieve import keywords


class OllamaBackend:
    def __init__(self, model, host="http://localhost:11434", temperature=0, num_ctx=16384, timeout=300):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.options = {"temperature": temperature, "num_ctx": num_ctx}

    def chat(self, system, user):
        r = requests.post(f"{self.host}/api/chat", timeout=self.timeout, json={
            "model": self.model, "stream": False, "format": "json", "options": self.options,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["message"]["content"]


class MockBackend:
    """NOT a real model. A keyword heuristic so the whole pipeline can be exercised without Ollama."""
    model = "mock-keyword-heuristic"

    def chat(self, system, user):
        test = re.search(r"^TEST: (.*)$", user, re.M).group(1)
        evidence = user.split("EVIDENCE START", 1)[-1]
        kws = {k.lower()[:5] for k in keywords(test)}
        best, best_score = "", 0.0
        for s in re.split(r"(?<=[.!?])\s+|\n", evidence):
            s = s.strip()
            if len(s) < 20 or s.startswith("[") and s.endswith("]"):
                continue
            toks = {t.lower()[:5] for t in re.findall(r"[A-Za-z0-9]+", s)}
            score = len(kws & toks) / max(len(kws), 1)
            if score > best_score:
                best, best_score = s, score
        if best_score >= 0.6:
            v, reason = "PASS", "MOCK: sentence shares most keywords with the test."
        elif best_score >= 0.3:
            v, reason = "UNCLEAR", "MOCK: only partial keyword overlap."
        else:
            return json.dumps({"verdict": "FAIL", "reason": "MOCK: no passage shares the test's keywords.",
                               "citations": [], "answer": None, "unclear_type": None, "review_flag": True})
        return json.dumps({"verdict": v, "reason": reason,
                           "citations": [{"document": "", "page": None, "quote": best}],
                           "answer": None, "unclear_type": None, "review_flag": True})


def make_backend(cfg):
    kind = cfg.get("backend", "ollama")
    if kind == "mock":
        return MockBackend()
    if kind == "ollama":
        return OllamaBackend(cfg["model"], cfg.get("host", "http://localhost:11434"),
                             cfg.get("temperature", 0), cfg.get("num_ctx", 16384), cfg.get("timeout_seconds", 300))
    raise ValueError(f"Unknown backend {kind!r}")
