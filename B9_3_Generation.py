"""
B9.3 - Danh gia BUOC SINH CAU TRA LOI (generation) voi 3 cau hinh, tren mau phan tang cua tap test:
  C0  LLM khong truy xuat   : chi co ma khach hang + cau hoi
  C1  RAG                   : them top-5 ban ghi truy xuat (tu CSDL)
  C2  RAG + cong tu choi    : neu chinh sach nguong (B9_2) tu choi thi KHONG goi LLM, tra "unknown"
Cau hoi dua cho LLM la cot Query day du; bo truy xuat nhan cot SearchText (cum da tach) nhu cac thi nghiem truoc.
LLM tra loi dang JSON {answer: yes|no|unknown, invoices:[...], explanation}; cham TU DONG (khong dung LLM lam giam khao):
  - Dung (retrieval)  : answer=yes, trich it nhat 1 hoa don thuoc tap dap an, khong trich hoa don khong ton tai / cua khach khac
  - Dung (tu choi)    : answer in {no, unknown}
  - Bia (fabricated)  : tra "yes" cho ca phai tu choi; hoac trich hoa don KHONG TON TAI; hoac hoa don cua KHACH KHAC (ro ri);
                        hoac "yes" nhung khong hoa don nao trich thuoc tap dap an (khang dinh khong duoc ho tro)
Dau ra: data/generation_B9.csv (tung ca x cau hinh) va bang tong hop co khoang tin cay + McNemar-Holm.
Chay:  set LLM_PROVIDER=... LLM_MODEL=... ; python B9_3_Generation.py   (them MOCK_LLM=1 de thu duong ong)
"""
import itertools
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crm_retrieval as R
import llm
import B9_2_Threshold as T

CASES_CSV = "data/test_cases_B8.csv"
POLICY_JSON = "data/policy_B9.json"
EXTRA_CSV = "data/test_cases_B9_extra.csv"      # nhom test bo sung cua B9_0 (neu co)
N_EXTRA_PER_MODULE = 15
OUT_CSV = "data/generation_B9.csv"
GEN_METHOD = None            # None: tu chon phuong phap co do chinh xac kiem tra cheo tren dev cao nhat trong policy_B9.json
N_RETR, N_ABS = 120, 60      # so ca truy xuat / ca phai tu choi lay mau tu tap test
SEED = 42
TOP_INV_CONTEXT = 5
MAX_LINES_PER_INVOICE = 3
CONDITIONS = ["C0_no_retrieval", "C1_rag", "C2_rag_gate"]

SYSTEM = ("You are a customer-support assistant for an online retailer. You answer questions about what a customer has "
          "purchased. Reply with ONE JSON object and nothing else: "
          '{"answer": "yes" | "no" | "unknown", "invoices": [invoice numbers you rely on], "explanation": "one or two sentences"}. '
          'Use "yes" only if the purchase is supported; use "no" if the records show the customer did not buy it; '
          'use "unknown" if you cannot tell. Never invent invoice numbers.')


# ------------------------------------------------------------------ mau
def stratified_sample(df, n, seed):
    rng = np.random.default_rng(seed)
    g = df.groupby("SubType")
    alloc = {k: max(1, round(n * len(v) / len(df))) for k, v in g}
    while sum(alloc.values()) > n:
        alloc[max(alloc, key=alloc.get)] -= 1
    while sum(alloc.values()) < n:
        alloc[max(alloc, key=alloc.get)] += 1
    parts = [v.sample(min(alloc[k], len(v)), random_state=int(rng.integers(1 << 30))) for k, v in g]
    return pd.concat(parts)


# ------------------------------------------------------------------ truy xuat + cong
def retrieve(ret, method, text, cust):
    if method.startswith("hyb_"):
        res = ret.run_all(text, cust, methods=["bm25", R.HYBRIDS[method]])
        return res[method][0]
    return ret.run_base(method, text, cust)[0]


def gate_accepts(policies, method, rows, ntok):
    if policies is None or method not in policies["methods"]:
        return True
    info = policies["methods"][method]
    pol = info["params"][info["chosen"]]
    d1, d2 = R.top_distances(rows)
    if not rows:
        d1 = d2 = 1.0
    D = dict(d1=np.array([d1]), d2=np.array([d2]), ntok=np.array([float(ntok)]), hyb=method.startswith("hyb_"),
             bm=np.array([rows[0].get("bm", 1.0) if rows else 1.0]), em=np.array([rows[0].get("emb", 1.0) if rows else 1.0]))
    return bool(T.apply_policy(pol, D)[0][0])


def fetch_records(ret, rows, cust):
    """Lay chi tiet dong giao dich cho top-5 hoa don duoc truy xuat (ten cot duoc do tu information_schema)."""
    inv = R.ranked_invoices(rows, TOP_INV_CONTEXT)
    if not inv:
        return []
    ret.cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'transactions';")
    cols = {r["column_name"] for r in ret.cur.fetchall()}
    extra = [c for c in ("InvoiceDate", "Quantity", "UnitPrice") if c in cols]
    sel = ", ".join(['"InvoiceNo"::text AS inv', '"CustomerID"::text AS cust', '"Description" AS dsc'] + [f'"{c}" AS "{c}"' for c in extra])
    ret.cur.execute(f'SELECT {sel} FROM transactions WHERE "InvoiceNo"::text = ANY(%s) AND "Description" = ANY(%s);',
                    (inv, list({r["Description"] for r in rows})))
    out, seen = [], {}
    order = {i: k for k, i in enumerate(inv)}
    for r in sorted(ret.cur.fetchall(), key=lambda r: order.get(r["inv"], 99)):
        if seen.get(r["inv"], 0) >= MAX_LINES_PER_INVOICE:
            continue
        seen[r["inv"]] = seen.get(r["inv"], 0) + 1
        out.append(r)
    return out


def fmt_records(recs):
    if not recs:
        return "(no matching records)"
    lines = []
    for r in recs:
        parts = [f"invoice {r['inv']}", f"customer {r['cust']}", r["dsc"]]
        for c, lab in (("InvoiceDate", "date"), ("Quantity", "qty"), ("UnitPrice", "unit price")):
            if c in r and r[c] is not None:
                parts.append(f"{lab} {r[c]}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def user_prompt(cust, query, records=None):
    s = f"Customer ID: {cust or 'not given'}\nQuestion: {query}\n"
    if records is not None:
        s += f"\nPurchase records retrieved from the database (may be incomplete or irrelevant):\n{records}\n\nAnswer using only these records."
    return s


# ------------------------------------------------------------------ cham diem
def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    ans = str(d.get("answer", "")).strip().lower()
    inv = d.get("invoices", [])
    inv = [str(x).strip() for x in (inv if isinstance(inv, list) else [inv]) if str(x).strip()]
    return (ans if ans in ("yes", "no", "unknown") else "unparsed"), inv, str(d.get("explanation", ""))[:300]


def lookup_owners(ret, invoices):
    if not invoices:
        return {}
    ret.cur.execute('SELECT "InvoiceNo"::text AS inv, "CustomerID"::text AS cust FROM transactions WHERE "InvoiceNo"::text = ANY(%s);', (list(invoices),))
    owners = {}
    for r in ret.cur.fetchall():
        owners.setdefault(r["inv"], set()).add(r["cust"])
    return owners


def score(answer, invoices, truth, cust, is_abstain, owners):
    nonexist = [i for i in invoices if i not in owners]
    foreign = [i for i in invoices if i in owners and cust and cust not in owners[i]]
    supported = [i for i in invoices if i in truth]
    if is_abstain:
        correct = answer in ("no", "unknown")
        unsupported_yes = answer == "yes"
    else:
        correct = answer == "yes" and len(supported) >= 1 and not nonexist and not foreign
        unsupported_yes = answer == "yes" and len(supported) == 0
    fabricated = bool(nonexist or foreign or unsupported_yes)
    return dict(Correct=float(correct), Fabricated=float(fabricated), NonexistentCite=float(bool(nonexist)),
                ForeignCite=float(bool(foreign)), UnsupportedYes=float(unsupported_yes),
                OverRefusal=float((not is_abstain) and answer in ("no", "unknown")))


# ------------------------------------------------------------------ chinh
def holm(p):
    o = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for r_, i in enumerate(o):
        run = max(run, (len(p) - r_) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    policies = json.load(open(POLICY_JSON)) if os.path.exists(POLICY_JSON) else None
    method = GEN_METHOD
    if method is None and policies:
        method = max(policies["methods"], key=lambda m: policies["methods"][m]["cv_dev"][policies["methods"][m]["chosen"]])
        print(f"Phuong phap truy xuat duoc chon (CV dev cao nhat): {method} ({policies['methods'][method]['chosen']})")
    method = method or "bm25"
    if policies is None:
        print("⚠️  Khong co policy_B9.json: cong tu choi (C2) se luon chap nhan => C2 == C1")

    allc = pd.read_csv(CASES_CSV, dtype=str, keep_default_na=False)
    test = allc[allc.get("Split", "test") == "test"] if "Split" in allc else allc
    sample = pd.concat([stratified_sample(test[test.EvalMode == "retrieval"], N_RETR, SEED),
                        stratified_sample(test[test.EvalMode == "abstain"], N_ABS, SEED)]).reset_index(drop=True)
    sample["Module"] = "core"
    if os.path.exists(EXTRA_CSV):
        ex = pd.read_csv(EXTRA_CSV, dtype=str, keep_default_na=False)
        ex = ex[ex.Split == "test"]
        ex_s = pd.concat([g.sample(min(N_EXTRA_PER_MODULE, len(g)), random_state=SEED) for _, g in ex.groupby("Module")])
        sample = pd.concat([sample, ex_s], ignore_index=True)
    print(f"Mau: {len(sample)} ca ({(sample.EvalMode == 'retrieval').sum()} truy xuat, {(sample.EvalMode == 'abstain').sum()} tu choi); "
          f"theo nhom: {sample.Module.value_counts().to_dict()}")

    ret = R.Retriever(load_models=method in ("mini", "bge", "e5") or method.startswith("hyb_"), need_bm25=True)
    rows_out = []
    for i, c in sample.iterrows():
        cust = c.CustomerID or None
        truth = set(filter(None, c.ExpectedInvoiceNos.split("|")))
        is_abs = c.EvalMode == "abstain"
        rows = retrieve(ret, method, c.SearchText, cust)
        ntok = len(R.tokenize(R.clean(c.SearchText)))
        accept = gate_accepts(policies, method, rows, ntok)
        recs = fetch_records(ret, rows, cust)
        prompts = {"C0_no_retrieval": user_prompt(cust, c.Query),
                   "C1_rag": user_prompt(cust, c.Query, fmt_records(recs))}
        for cond in CONDITIONS:
            if cond == "C2_rag_gate" and not accept:
                raw, ans, inv, expl = "(gate: refused, no LLM call)", "unknown", [], "gate"
            else:
                raw = llm.chat(SYSTEM, prompts["C1_rag" if cond != "C0_no_retrieval" else "C0_no_retrieval"], json_mode=True)
                ans, inv, expl = parse(raw)
            sc = score(ans, inv, truth, cust, is_abs, lookup_owners(ret, inv))
            rows_out.append(dict(CaseID=c.QueryID, Module=c.Module, ContextHasTruth=float(any(r_["inv"] in truth for r_ in recs)) if recs else 0.0, ContextN=len(recs), Context=fmt_records(recs)[:700], Type=c.Type, SubType=c.SubType, EvalMode=c.EvalMode, Condition=cond,
                                 Method=method, GateAccepted=float(accept), Answer=ans, Invoices="|".join(inv),
                                 Explanation=expl, **sc))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sample)}")
    ret.close()

    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_CSV, index=False)
    rng = np.random.default_rng(SEED)

    def ci(x):
        x = np.asarray(x, float)
        m = x[rng.integers(0, len(x), (5000, len(x)))].mean(1)
        return np.percentile(m, [2.5, 97.5])
    print("\n=== KET QUA (95% CI bootstrap) ===")
    tab = []
    for cond in CONDITIONS:
        d = df[df.Condition == cond]
        for grp, sub in (("Tat ca", d), ("Truy xuat", d[d.EvalMode == "retrieval"]), ("Phai tu choi", d[d.EvalMode == "abstain"])):
            for met in ("Correct", "Fabricated", "OverRefusal"):
                if grp == "Phai tu choi" and met == "OverRefusal":
                    continue
                lo, hi = ci(sub[met].values)
                tab.append(dict(Condition=cond, Nhom=grp, ChiSo=met, n=len(sub), TB=sub[met].mean(), lo=lo, hi=hi))
    tab = pd.DataFrame(tab)
    print(tab.pivot_table(index=["Nhom", "ChiSo"], columns="Condition", values="TB").round(3).to_string())
    tab.to_csv("data/generation_summary_B9.csv", index=False)
    print("\n=== THEO NHOM (Correct / Fabricated) ===")
    print(df.pivot_table(index="Module", columns="Condition", values=["Correct", "Fabricated"]).round(3).to_string())
    c1 = df[(df.Condition == "C1_rag") & (df.EvalMode == "retrieval")]
    print("\n=== CHAN DOAN BUOC SINH (C1, ca truy xuat): ngu canh co chua hoa don dung khong? ===")
    print(f"  ngu canh chua hoa don dung: {c1.ContextHasTruth.mean():.0%} | LLM tra loi dung khi ngu canh co dap an: "
          f"{c1[c1.ContextHasTruth == 1].Correct.mean():.0%} | khong doc duoc JSON (unparsed): {(df.Answer == 'unparsed').mean():.0%}")
    print("  (ngu canh ~100% ma LLM dung thap => mo hinh yeu / prompt can doi; ngu canh thap => loi lay ban ghi, xem cot Context)")
    print("\n=== Phan tich chi tiet loi bia (ty le trong tat ca ca) ===")
    print(df.groupby("Condition")[["NonexistentCite", "ForeignCite", "UnsupportedYes"]].mean().round(3).to_string())
    for met in ("Correct", "Fabricated"):
        w = df.pivot(index="CaseID", columns="Condition", values=met)
        prs = list(itertools.combinations(CONDITIONS, 2))
        ps = []
        for a, b in prs:
            n10, n01 = int(((w[a] == 1) & (w[b] == 0)).sum()), int(((w[a] == 0) & (w[b] == 1)).sum())
            ps.append(stats.binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0)
        print(f"\nMcNemar-Holm cho {met}:")
        for (a, b), p, q in zip(prs, ps, holm(np.array(ps))):
            print(f"  {a} vs {b}: chenh={w[a].mean() - w[b].mean():+.3f}, p_Holm={q:.4f}")
    print(f"\n✅ Da luu {OUT_CSV}, data/generation_summary_B9.csv")


if __name__ == "__main__":
    main()
