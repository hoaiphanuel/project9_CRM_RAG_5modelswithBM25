"""Shared helpers for the RAG pilot steps (B7_4 .. B7_7). Not run directly."""
import json, math, os, random, re, sys, time
from collections import Counter

import numpy as np
import pandas as pd

DATA_DIR = os.getenv("B7_DATA_DIR", "data")
P = lambda f: os.path.join(DATA_DIR, f)
SEED = 42
TOP_K = 5
CONDITIONS = ["llm_only", "rag_bm25", "rag_hybrid"]
BM25_K1, BM25_B = 1.5, 0.75
norm = lambda s: str(s).strip().lower() if s is not None else ""
tokenize = lambda t: re.findall(r"[a-z0-9]+", str(t).lower())

POS_QUOTA = {"exact": 3, "case_variant": 3, "partial_phrase": 4, "keyword_generic": 3, "multi_product": 3, "noisy_query": 3,
             "spacing_noise": 3, "typo": 3, "typo_keyboard": 3, "word_shuffle": 2, "attribute_color_noun": 5, "conversational": 5,
             "paraphrase_synonym": 5, "filter_qty": 5, "filter_invoice_value": 5, "filter_date_range": 5,
             "filter_last_6_months": 5, "filter_combo": 5}                                   # = 70 answerable
NEG_QUOTA = {"neg_not_bought_similar": 10, "neg_not_bought_random": 6, "neg_nonexistent_product": 4, "neg_unknown_customer": 4}
NEG_COMPLEX_EXTRA = 6                                                                          # filter cases with no valid answer -> 30

SYSTEM_RAG = ("You are a customer-support assistant for an online retailer. Answer the agent's question using ONLY the purchase records "
              "provided below. If the records do not contain the answer, say clearly that you could not find it. Never guess or add "
              "information that is not in the records. Answer in one or two sentences and cite product names and invoice numbers "
              "when you use them.")
SYSTEM_NONE = ("You are a customer-support assistant for an online retailer. Answer the agent's question. No purchase records are "
               "available to you. If you cannot know the answer, say clearly that you could not find it. Never guess. "
               "Answer in one or two sentences.")


def user_prompt(question, context):
    return f"Question: {question}\n\nPurchase records:\n{context}" if context is not None else f"Question: {question}"


# ------------------------------------------------------------------ LLM backends
def call_llm(system, user):
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    model = os.getenv("LLM_MODEL", "mock")
    if provider == "mock":
        m = re.search(r"\n1\. (.+?) \|", user)
        if "No purchase records" in system or "could not find it" in user.split("Purchase records:")[-1][:200] or not m:
            return "I could not find that in the records."
        inv = re.search(r"INV (\d+)", user)
        return f"Yes, the customer bought {m.group(1)} (invoice {inv.group(1) if inv else 'n/a'})."
    import requests
    for attempt in range(5):
        try:
            if provider == "anthropic":
                r = requests.post("https://api.anthropic.com/v1/messages", timeout=120,
                                  headers={"x-api-key": os.environ["LLM_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
                                  json={"model": model, "max_tokens": 250, "temperature": 0, "system": system,
                                        "messages": [{"role": "user", "content": user}]})
                r.raise_for_status(); return r.json()["content"][0]["text"].strip()
            base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            hdr = {"content-type": "application/json"}
            if os.getenv("LLM_API_KEY"): hdr["authorization"] = "Bearer " + os.environ["LLM_API_KEY"]
            r = requests.post(base + "/chat/completions", headers=hdr, timeout=180,
                              json={"model": model, "temperature": 0, "max_tokens": 250,
                                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            r.raise_for_status(); return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("   LLM error, retrying:", str(e)[:120]); time.sleep(2 * (attempt + 1))
    return "[LLM ERROR]"


def load_ctx():
    return [json.loads(l) for l in open(P("rag_contexts_B7.jsonl"), encoding="utf8")]


# ------------------------------------------------------------------ automatic checks (screening only)
ABSTAIN_RE = re.compile(r"(could not find|couldn't find|can(?:no|')t find|cannot find|no (?:matching |relevant )?(?:record|purchase)|not found|"
                        r"do(?:es)? not (?:appear|have|show)|don't have|no information|unable to|not available|did not (?:buy|purchase)|has not (?:bought|purchased))", re.I)


def auto_checks(case, cond, ans):
    abst = bool(ABSTAIN_RE.search(ans))
    gt = [norm(x) for x in case["GroundTruth"]]
    a = norm(ans)
    if case["ShouldAbstain"]: auto_correct = abst
    else: auto_correct = (not abst) and any(g in a for g in gt)
    ctx = case["contexts"][cond]["text"] if cond != "llm_only" else ""
    allowed = set(re.findall(r"\d{5,7}", ctx + " " + case["Question"]))
    unsupported_num = any(n not in allowed for n in re.findall(r"\d{5,7}", ans))
    return abst, auto_correct, unsupported_num


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int); po = (a == b).mean()
    pe = sum(((a == k).mean() * (b == k).mean()) for k in (0, 1)); return (po - pe) / (1 - pe) if pe < 1 else 1.0


def mcnemar_p(b, c):
    n = b + c; return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)




def banner(step, title, reads, writes, nxt):
    print("=" * 74); print(f"BƯỚC {step}: {title}")
    print("  đọc  :", "; ".join(reads)); print("  ghi  :", "; ".join(writes)); print("  tiếp :", nxt); print("=" * 74)

