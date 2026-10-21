"""
llm.py - Goi mo hinh ngon ngu qua HTTP (khong can SDK), co bo nho dem tren dia de chay lai khong ton chi phi.

Cau hinh (KHONG can dat bien moi truong): sua file llm_config.py cung thu muc (xem mau). Bien moi truong, neu co, duoc uu tien hon.
  LLM_PROVIDER = gemini | anthropic | openai | ollama      (mac dinh: anthropic)
  LLM_MODEL    = ten mo hinh (BAT BUOC)
  gemini    : GEMINI_API_KEY   (REST generateContent cua Google AI Studio; tuy chon GEMINI_BASE_URL)
  anthropic : ANTHROPIC_API_KEY   (tuy chon ANTHROPIC_BASE_URL)
  openai    : OPENAI_API_KEY      (tuy chon OPENAI_BASE_URL, dung cho moi may chu tuong thich OpenAI)
  ollama    : OLLAMA_URL          (mac dinh http://localhost:11434)
  LLM_MIN_INTERVAL = so giay toi thieu giua 2 lan goi (de tranh loi 429 cua goi mien phi; vi du 4)
  MOCK_LLM=1 : tra loi gia (de kiem tra duong ong ma khong goi mo hinh)
"""
import hashlib
import json
import os
import time

import requests

# ---- doc llm_config.py (neu co) de khong can dat bien moi truong ngoai chuong trinh
try:
    import llm_config as _cfg
    for _k in ("LLM_PROVIDER", "LLM_MODEL", "LLM_MIN_INTERVAL", "GEMINI_API_KEY", "GEMINI_BASE_URL", "ANTHROPIC_API_KEY",
               "OPENAI_API_KEY", "OPENAI_BASE_URL", "OLLAMA_URL", "LLM_CACHE"):
        if hasattr(_cfg, _k) and not os.getenv(_k) and str(getattr(_cfg, _k)).strip():
            os.environ[_k] = str(getattr(_cfg, _k))
except ImportError:
    pass

PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
MODEL = os.getenv("LLM_MODEL", "")
MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "0") or 0)
_last_call = [0.0]
CACHE_PATH = os.getenv("LLM_CACHE", "data/llm_cache.jsonl")
_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if os.path.exists(CACHE_PATH):
            for line in open(CACHE_PATH, encoding="utf8"):
                try:
                    d = json.loads(line)
                    _cache[d["k"]] = d["v"]
                except Exception:       # noqa: BLE001
                    pass
    return _cache


def _post(url, headers, payload, tries=6):
    for i in range(tries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError("LLM request failed")


def chat(system, user, max_tokens=400, temperature=0.0, json_mode=False):
    if os.getenv("MOCK_LLM") == "1":
        return '{"answer": "unknown", "invoices": [], "explanation": "mock"}'
    if not MODEL:
        raise SystemExit("Hay dat bien moi truong LLM_MODEL (va LLM_PROVIDER, khoa API neu can).")
    key = hashlib.sha256(json.dumps([PROVIDER, MODEL, system, user, temperature, max_tokens]).encode()).hexdigest()
    cache = _load_cache()
    if key in cache:
        return cache[key]
    wait = MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()
    if PROVIDER == "gemini":
        key_ = os.environ.get("GEMINI_API_KEY", "")
        if not key_ or key_.startswith("PASTE"):
            raise SystemExit("Chua co GEMINI_API_KEY: dien khoa vao llm_config.py (khong chia se file nay).")
        base = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        gen = {"temperature": temperature, "maxOutputTokens": max(max_tokens, 512)}     # chua cho cho phan "suy nghi" (neu co)
        if "2.5-flash" in MODEL:                       # goi Flash 2.5 co the tat suy nghi de khong an het ngan sach dau ra
            gen["thinkingConfig"] = {"thinkingBudget": 0}
        j = _post(f"{base}/models/{MODEL}:generateContent", {"x-goog-api-key": key_, "Content-Type": "application/json"},
                  {"systemInstruction": {"parts": [{"text": system}]},
                   "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": gen})
        cands = j.get("candidates") or []
        parts = (cands[0].get("content", {}).get("parts", []) if cands else [])
        out = "".join(p.get("text", "") for p in parts)
        if not out:
            out = ""        # bi chan hoac rong: tra chuoi rong de script xu ly (parse -> 'unparsed')
    elif PROVIDER == "anthropic":
        base = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        j = _post(f"{base}/v1/messages",
                  {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
                  {"model": MODEL, "max_tokens": max_tokens, "temperature": temperature, "system": system,
                   "messages": [{"role": "user", "content": user}]})
        out = "".join(b.get("text", "") for b in j["content"] if b.get("type") == "text")
    elif PROVIDER == "openai":
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        j = _post(f"{base}/chat/completions", {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
                  {"model": MODEL, "temperature": temperature, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        out = j["choices"][0]["message"]["content"]
    elif PROVIDER == "ollama":
        base = os.getenv("OLLAMA_URL", "http://localhost:11434")
        j = _post(f"{base}/api/chat", {}, {"model": MODEL, "stream": False, "options": {"temperature": temperature},
                                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        out = j["message"]["content"]
    else:
        raise SystemExit(f"LLM_PROVIDER khong ho tro: {PROVIDER}")
    os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
    with open(CACHE_PATH, "a", encoding="utf8") as f:
        f.write(json.dumps({"k": key, "v": out}, ensure_ascii=False) + "\n")
    cache[key] = out
    return out
