"""
B8.1 (v4) - Chay thu nghiem 8 phuong phap trong CUNG 1 lan tren bo test B8.0:
  rule (ILIKE), bm25, mini, bge, e5, hyb_mini, hyb_bge, hyb_e5   (hyb_* = BM25 ket hop embedding bang RRF)

Tat ca phuong phap co so deu chay bang cau SQL tham so hoa trong PostgreSQL (BM25 dung cac bang chi so bm25_* tu dung o muc [3b]).
Hybrid: hop nhat xep hang BM25 va embedding bang RRF (k=60) o muc mo ta san pham; hai truy van van chay trong PostgreSQL,
buoc cong diem 1/(k+hang) lam o ung dung; thoi gian hybrid = tong thoi gian hai truy van. Khoang cach hybrid = min(d_bm25, d_embedding)
cua mo ta xep hang 1 (thieu = 1.0). BM25: d = 1/(1+diem). Moi phuong phap co nguong TAU rieng chon tren dev (Youden J).
File log cu (data/test_results_B8.csv) bi XOA va ghi lai tu dau: hay sao luu truoc khi chay.
Cac script thong ke/ve hinh di kem: B8_2_thongke_BM25.py, B8_3_Plotdiagram_BM25.py (da cap nhat cho 8 phuong phap).
"""
import os
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from scipy.stats import mannwhitneyu
from sentence_transformers import SentenceTransformer

# =====================================================
# CAU HINH
# =====================================================
CASES_CSV = "data/test_cases_B8.csv"
LOG_CSV = "data/test_results_B8.csv"
SUMMARY_CSV = "data/summary_B8.csv"

K_LIST = [1, 3, 5, 10]       # bao cao Hit@K cho tung K
K_MAIN = 5                   # K dung de tinh "Success" chinh (phai nam trong K_LIST)
FETCH = 60                   # lay nhieu dong roi loai trung hoa don
TAU_DEFAULT = 0.5            # chi dung khi khong co tap dev
AUTO_TAU = True              # chon nguong tu choi tren tap dev
NORMALIZE_TEXT = True        # bo khoang trang thua o cum tim (ap dung cho ca 4 phuong phap)
RUN_SECURITY_TEST = True
RRF_K = 60                   # hang so RRF cho cac phuong phap hybrid
BM25_K1, BM25_B = 1.5, 0.75  # tham so BM25 mac dinh (Okapi)
REBUILD_BM25 = False         # True: dung lai cac bang chi so BM25 (vd khi bang transactions thay doi)
BOOTSTRAP = 2000

assert K_MAIN in K_LIST
DB = dict(dbname=os.getenv("PGDATABASE", "retaildb"), user=os.getenv("PGUSER", "postgres"),
          password=os.getenv("PGPASSWORD", "postgres"), host=os.getenv("PGHOST", "localhost"),
          port=os.getenv("PGPORT", "5432"))

HIT_COLS = [f"Hit@{k}" for k in K_LIST]
COLS = (["CaseID", "Split", "Type", "SubType", "EvalMode", "Model", "NumExpected", "EncodeTime", "DBTime",
         "ResponseTime", "Top1Distance", "Accepted"] + HIT_COLS +
        ["MRR", "Precision@k", "Success", "AbstainOK", "Hallucination", "InjectionSafe",
         "SecurityViolation", "Top1Text"])


def save_rows(rows):
    """Ghi noi tiep vao file log ngay sau moi ca."""
    pd.DataFrame(rows).reindex(columns=COLS).to_csv(
        LOG_CSV, mode="a", header=not os.path.exists(LOG_CSV), index=False)


# =====================================================
# [1] NAP TEST CASE
# =====================================================
print(">>> [1] Nap test case...")
allc = pd.read_csv(CASES_CSV, dtype=str, keep_default_na=False)
if "Split" not in allc.columns:
    allc["Split"] = "test"
dev = allc[allc.Split == "dev"].reset_index(drop=True)
cases = allc[allc.Split == "test"].reset_index(drop=True)
print(f"✅ test={len(cases)} ca ({cases.Type.value_counts().to_dict()}), dev={len(dev)} ca")

# =====================================================
# [2] KET NOI POSTGRESQL
# =====================================================
print(">>> [2] Ket noi PostgreSQL...")
conn = psycopg2.connect(**DB)
conn.autocommit = True
cur = conn.cursor(cursor_factory=RealDictCursor)
print("✅ Da ket noi")

# =====================================================
# [3] NAP MO HINH EMBEDDING
# (model, cot embedding, tien to cau truy van). Tien to phai khop cach nhung tai lieu
# luc lap chi muc (vd E5: truy van "query: ", tai lieu "passage: ").
# =====================================================
EMB_SPEC = {"mini": ("sentence-transformers/all-MiniLM-L6-v2", "embedding1", ""),
            "bge":  ("BAAI/bge-large-en-v1.5", "embedding2", ""),
            "e5":   ("intfloat/e5-large-v2", "embedding3", "")}
HYBRIDS = {"hyb_mini": "mini", "hyb_bge": "bge", "hyb_e5": "e5"}   # hybrid = BM25 + embedding tuong ung
BASE_METHODS = ["rule", "bm25"] + list(EMB_SPEC)
ALL_METHODS = BASE_METHODS + list(HYBRIDS)                 # thu tu hien thi
RUN = ALL_METHODS                                          # chay tat ca trong cung 1 lan
DIST_METHODS = ["bm25"] + list(EMB_SPEC) + list(HYBRIDS)   # phuong phap co nguong TAU / khoang cach
MODELS = {}
print(">>> [3] Nap mo hinh embedding...")
for k, (ckpt, col, pre) in EMB_SPEC.items():
    MODELS[k] = (SentenceTransformer(ckpt), col, pre)
print("✅ Da nap: " + ", ".join(MODELS))
METHODS = RUN


# =====================================================
# [3b] BM25 CHAY TRONG POSTGRESQL (SQL tham so hoa, giong cac phuong phap con lai)
# Chi so BM25 = 4 bang phu tao 1 lan (giong 1 chi so nguoc):
#   bm25_docs(doc_id, description, dl)  : moi mo ta san pham duy nhat la 1 tai lieu, dl = so token
#   bm25_terms(term, doc_id, tf)        : tan suat tu trong tai lieu (co chi so theo term)
#   bm25_idf(term, df, idf)             : idf = ln(1 + (N - df + 0.5) / (df + 0.5))
#   bm25_stats(n_docs, avgdl)
# Token hoa: chu thuong, tach theo [^a-z0-9]+ (giong tokenize() ben duoi).
# =====================================================
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN.findall(text.lower())


BM25_SETUP = [
    "DROP TABLE IF EXISTS bm25_stats, bm25_idf, bm25_terms, bm25_docs;",
    """CREATE TABLE bm25_docs AS
       SELECT row_number() OVER (ORDER BY description)::int AS doc_id, description
       FROM (SELECT DISTINCT "Description" AS description FROM transactions
             WHERE "Description" IS NOT NULL) t;""",
    "ALTER TABLE bm25_docs ADD PRIMARY KEY (doc_id);",
    """CREATE TABLE bm25_terms AS
       SELECT d.doc_id, w.term, COUNT(*)::int AS tf
       FROM bm25_docs d
       CROSS JOIN LATERAL regexp_split_to_table(lower(d.description), '[^a-z0-9]+') AS w(term)
       WHERE w.term <> ''
       GROUP BY d.doc_id, w.term;""",
    "CREATE INDEX bm25_terms_term_idx ON bm25_terms (term);",
    "CREATE INDEX bm25_terms_doc_idx ON bm25_terms (doc_id);",
    "ALTER TABLE bm25_docs ADD COLUMN dl int;",
    "UPDATE bm25_docs d SET dl = x.dl FROM (SELECT doc_id, SUM(tf)::int AS dl FROM bm25_terms GROUP BY doc_id) x WHERE x.doc_id = d.doc_id;",
    "UPDATE bm25_docs SET dl = 0 WHERE dl IS NULL;",
    """CREATE TABLE bm25_stats AS SELECT COUNT(*)::float8 AS n_docs, AVG(dl)::float8 AS avgdl FROM bm25_docs;""",
    """CREATE TABLE bm25_idf AS
       SELECT t.term, COUNT(*)::int AS df,
              LN(1 + (s.n_docs - COUNT(*) + 0.5) / (COUNT(*) + 0.5))::float8 AS idf
       FROM bm25_terms t, bm25_stats s GROUP BY t.term, s.n_docs;""",
    "ALTER TABLE bm25_idf ADD PRIMARY KEY (term);",
    # chi so tren transactions de noi mo ta -> hoa don nhanh (khong doi du lieu, chi them chi so)
    'CREATE INDEX IF NOT EXISTS transactions_desc_idx ON transactions ("Description");',
    "ANALYZE bm25_docs; ANALYZE bm25_terms; ANALYZE bm25_idf;",
]


def ensure_bm25_index():
    cur.execute("SELECT to_regclass('bm25_idf') AS t;")
    exists = cur.fetchone()["t"] is not None
    if exists and not REBUILD_BM25:
        print("✅ Dung lai chi so BM25 da co (REBUILD_BM25 = False)")
    else:
        print(">>> [3b] Dung chi so BM25 trong PostgreSQL (1 lan)...")
        t0 = time.perf_counter()
        for q in BM25_SETUP:
            cur.execute(q)
        print(f"✅ Xong sau {time.perf_counter() - t0:.1f}s")
    cur.execute("SELECT n_docs, avgdl FROM bm25_stats;")
    st = cur.fetchone()
    cur.execute("SELECT COUNT(*) AS n FROM bm25_idf;")
    print(f"   BM25: {int(st['n_docs']):,} mo ta duy nhat, avgdl={st['avgdl']:.2f}, {cur.fetchone()['n']:,} token")


if "bm25" in RUN:
    ensure_bm25_index()

BM25_SQL = """
WITH sc AS (
    SELECT t.doc_id,
           SUM(i.idf * t.tf * ({k1} + 1) / (t.tf + {k1} * (1 - {b} + {b} * d.dl / s.avgdl))) AS score
    FROM bm25_terms t
    JOIN bm25_idf  i ON i.term = t.term
    JOIN bm25_docs d ON d.doc_id = t.doc_id
    CROSS JOIN bm25_stats s
    WHERE t.term = ANY(%s)
    GROUP BY t.doc_id
)
SELECT tr."InvoiceNo", tr."Description", 1.0 / (1.0 + sc.score) AS distance
FROM sc
JOIN bm25_docs d ON d.doc_id = sc.doc_id
JOIN transactions tr ON tr."Description" = d.description
{where}
ORDER BY sc.score DESC, d.doc_id, tr."InvoiceNo"
LIMIT %s;
""".replace("{k1}", repr(float(BM25_K1))).replace("{b}", repr(float(BM25_B)))


def bm25_search(text, cust):
    t0 = time.perf_counter()
    terms = sorted(set(tokenize(text)))              # tu lap trong truy van chi tinh 1 lan
    t_enc = time.perf_counter() - t0
    where = 'WHERE tr."CustomerID"::text = %s' if cust else ""
    params = [terms] + ([cust] if cust else []) + [FETCH]
    t1 = time.perf_counter()
    cur.execute(BM25_SQL.replace("{where}", where), params)   # chuoi nguoi dung chi di vao tham so rang buoc
    rows = cur.fetchall()
    return rows, t_enc, time.perf_counter() - t1


# =====================================================
# [4] TIM KIEM
# =====================================================
def clean(text):
    return " ".join(text.split()) if NORMALIZE_TEXT else text


def rule_based(text, cust):
    where = 'AND "CustomerID"::text = %s' if cust else ""
    params = [f"%{text}%"] + ([cust] if cust else []) + [FETCH]
    t0 = time.perf_counter()
    cur.execute(f"""SELECT "InvoiceNo","Description" FROM transactions
                    WHERE "Description" ILIKE %s {where} LIMIT %s;""", params)
    rows = cur.fetchall()
    return rows, 0.0, time.perf_counter() - t0


def semantic(text, cust, name):
    model, col, prefix = MODELS[name]
    t0 = time.perf_counter()
    q = model.encode(prefix + text, normalize_embeddings=True).tolist()
    t_enc = time.perf_counter() - t0
    where = 'WHERE "CustomerID"::text = %s' if cust else ""
    params = [q] + ([cust] if cust else []) + [q, FETCH]
    t1 = time.perf_counter()
    cur.execute(f"""SELECT "InvoiceNo","Description", {col} <=> %s::vector AS distance
                    FROM transactions {where}
                    ORDER BY {col} <=> %s::vector LIMIT %s;""", params)
    rows = cur.fetchall()
    return rows, t_enc, time.perf_counter() - t1


def run_method(name, text, cust):
    text = clean(text)
    if name == "rule":
        return rule_based(text, cust)
    if name == "bm25":
        return bm25_search(text, cust)
    return semantic(text, cust, name)


def rrf_fuse(bm_rows, emb_rows, k=RRF_K, limit=FETCH):
    """Hop nhat 2 danh sach dong bang RRF o muc mo ta san pham (xem docstring)."""
    def ranks(rows):
        r = {}
        for x in rows:
            r.setdefault(x["Description"], len(r) + 1)
        return r
    rb, re_ = ranks(bm_rows), ranks(emb_rows)
    dist_b, dist_e, inv = {}, {}, {}
    for src, store in ((bm_rows, dist_b), (emb_rows, dist_e)):
        for x in src:
            store.setdefault(x["Description"], float(x["distance"]))
    for x in list(bm_rows) + list(emb_rows):
        inv.setdefault(x["Description"], [])
        if x["InvoiceNo"] not in inv[x["Description"]]:
            inv[x["Description"]].append(x["InvoiceNo"])
    score_ = {d: (1.0 / (k + rb[d]) if d in rb else 0.0) + (1.0 / (k + re_[d]) if d in re_ else 0.0) for d in set(rb) | set(re_)}
    order_ = sorted(score_, key=lambda d: (-score_[d], min(rb.get(d, 10 ** 6), re_.get(d, 10 ** 6)), d))
    out = []
    for d in order_:
        dist = min(dist_b.get(d, 1.0), dist_e.get(d, 1.0))
        for i in inv[d]:
            out.append({"InvoiceNo": i, "Description": d, "distance": dist})
        if len(out) >= limit:
            break
    return out[:limit]


def run_case(text, cust, include_rule=True):
    """Chay moi phuong phap co ban 1 lan, suy ra hybrid tu ket qua da lay. Tra ve {ten: (rows, t_enc, t_db)}."""
    res = {m: run_method(m, text, cust) for m in BASE_METHODS if include_rule or m != "rule"}
    for h, m in HYBRIDS.items():
        res[h] = (rrf_fuse(res["bm25"][0], res[m][0]), res[m][1] + res["bm25"][1], res[m][2] + res["bm25"][2])
    return res


# =====================================================
# [5] CHI SO
# =====================================================
def ranked_invoices(rows):
    return list(dict.fromkeys(str(r["InvoiceNo"]) for r in rows))[:max(K_LIST)]


def score(ranked, truth):
    out = {f"Hit@{k}": float(any(x in truth for x in ranked[:k])) for k in K_LIST}
    first = next((i for i, x in enumerate(ranked) if x in truth), None)
    out["MRR"] = 1 / (first + 1) if first is not None else 0.0
    top = ranked[:K_MAIN]
    out["Precision@k"] = sum(x in truth for x in top) / len(top) if top else 0.0
    out["Success"] = out[f"Hit@{K_MAIN}"]            # "tim trong tap hop, co ket qua dung la dat"
    return out


def accepted(name, rows, tau):
    if not rows:
        return False
    return True if name == "rule" else float(rows[0]["distance"]) <= tau[name]


INJ = re.compile(r"(;\s*--)|(\b(DROP|DELETE|TRUNCATE)\b\s+(TABLE|FROM|DATABASE))|"
                 r"(\bUPDATE\b\s+\w+\s+\bSET\b)|(\bINSERT\b\s+\bINTO\b)|(\bUNION\b\s+\bSELECT\b)", re.I)

# =====================================================
# [6] KHOI DONG + CHON NGUONG TU CHOI TREN TAP DEV
# =====================================================
print(">>> [6] Khoi dong may (khong tinh gio)...")
run_case("warm up", None)

TAU_METHODS = [m for m in DIST_METHODS if m in RUN]
TAU = {m: TAU_DEFAULT for m in TAU_METHODS}
if AUTO_TAU and len(dev):
    print(">>> Chon nguong tu choi TAU tren tap dev (khong dung tap test)...")
    dev_d = {m: ([], []) for m in TAU_METHODS}                     # (ca tra loi duoc, ca phai tu choi)
    for _, c in dev.iterrows():
        res = run_case(c.SearchText, c.CustomerID or None, include_rule=False)
        for name in TAU_METHODS:
            rows = res[name][0]
            if rows:
                dist = float(rows[0]["distance"])
            elif name == "bm25" or name in HYBRIDS:
                dist = 1.0                      # khong khop token nao = khoang cach toi da
            else:
                continue
            dev_d[name][1 if c.EvalMode == "abstain" else 0].append(dist)
    for name in TAU_METHODS:
        d_ans, d_abs = np.array(dev_d[name][0]), np.array(dev_d[name][1])
        if len(d_ans) and len(d_abs):
            allv = np.unique(np.concatenate([d_ans, d_abs]))
            cands = np.concatenate([[allv[0] - 1e-6], (allv[:-1] + allv[1:]) / 2, [allv[-1] + 1e-6]])
            best = max(cands, key=lambda t: np.mean(d_ans <= t) - np.mean(d_abs <= t))   # Youden J
            TAU[name] = float(best)
            print(f"   {name:9s} TAU={best:.3f} | giu lai ca tra loi duoc: {np.mean(d_ans <= best):.0%} | "
                  f"nhan nham ca phai tu choi: {np.mean(d_abs <= best):.0%}")
else:
    print(f"   Dung TAU mac dinh = {TAU_DEFAULT}")

# luu TAU de bao cao (ghep voi file cu neu chi chay BM25)
TAU_CSV = "data/tau_B8.csv"
old_tau = pd.read_csv(TAU_CSV, index_col=0).Tau.to_dict() if os.path.exists(TAU_CSV) else {}
old_tau.update(TAU)
pd.DataFrame({"Tau": old_tau}).to_csv(TAU_CSV)
print(f"   Da luu nguong TAU vao {TAU_CSV}: " + ", ".join(f"{k}={v:.3f}" for k, v in old_tau.items()))

# =====================================================
# [7] CHAY TUNG CA (in ket qua, tich luy, ghi log ngay)
# =====================================================
print("\n>>> [7] Chay test...")
if os.path.exists(LOG_CSV):
    os.remove(LOG_CSV)
run_succ = {m: [] for m in METHODS}
run_abs = {m: [] for m in METHODS}
n_total = len(cases)
interrupted = False

try:
    for idx, c in cases.iterrows():
        cust = c.CustomerID or None
        truth = set(filter(None, c.ExpectedInvoiceNos.split("|")))
        flagged = bool(INJ.search(c.Query) or INJ.search(c.SearchText))
        print(f"\n=== [CA {idx + 1}/{n_total}] {c.QueryID} | {c.Type}/{c.SubType} | {c.EvalMode} | "
              f"CustomerID={cust or '-'} | Tim: \"{clean(c.SearchText)[:70]}\" | So hoa don dung={len(truth)} ===")

        case_rows = []
        results = run_case(c.SearchText, cust)
        for name in METHODS:
            rows, t_enc, t_db = results[name]
            ok_answer = accepted(name, rows, TAU)
            row = {"CaseID": c.QueryID, "Split": "test", "Type": c.Type, "SubType": c.SubType,
                   "EvalMode": c.EvalMode, "Model": name, "NumExpected": len(truth), "EncodeTime": t_enc,
                   "DBTime": t_db, "ResponseTime": t_enc + t_db, "Accepted": float(ok_answer),
                   "SecurityViolation": float(flagged),
                   "Top1Text": (rows[0]["Description"] if rows else "")[:60]}
            dist_txt = ""
            if name != "rule" and rows:
                row["Top1Distance"] = float(rows[0]["distance"])
                dist_txt = f", Distance={row['Top1Distance']:.3f}"
            elif name == "bm25" or name in HYBRIDS:   # khong khop token nao: khoang cach = 1.0 (de AUROC day du)
                row["Top1Distance"] = 1.0
                dist_txt = ", Distance=1.000"

            if c.EvalMode == "abstain":
                row["AbstainOK"] = float(not ok_answer)
                row["Hallucination"] = float(ok_answer)       # tra loi khi dang le phai tu choi
                row["Success"] = row["AbstainOK"]
                run_abs[name].append(row["AbstainOK"])
                print(f"[{name.upper():9s}] {'DAT ' if row['Success'] else 'SAI '} AbstainOK={bool(row['AbstainOK'])}"
                      f"{dist_txt}, Time={row['ResponseTime']:.3f}s")
            else:
                row.update(score(ranked_invoices(rows), truth))
                print(f"[{name.upper():9s}] {'DAT ' if row['Success'] else 'SAI '} Hit@1={bool(row['Hit@1'])}, "
                      f"Hit@{K_MAIN}={bool(row['Success'])}, MRR={row['MRR']:.2f}{dist_txt}, "
                      f"Time={row['ResponseTime']:.3f}s")
            run_succ[name].append(row["Success"])
            case_rows.append(row)

        save_rows(case_rows)
        print("   >> Ty le dat tich luy: " + " | ".join(f"{m} {np.mean(v):.2f}" for m, v in run_succ.items() if v))
except KeyboardInterrupt:
    interrupted = True
    print("\n⛔ Dung giua chung. Cac ca da chay van nam trong file log; tong hop tren phan da chay.")

# =====================================================
# [8] KIEM TRA SQL INJECTION (tuy chon)
# =====================================================
if RUN_SECURITY_TEST and not interrupted:
    print("\n>>> [8] Kiem tra SQL injection...")
    ATTACKS = ["'; DROP TABLE transactions; --", "x' OR '1'='1", "1; DELETE FROM transactions WHERE 1=1; --",
               "' UNION SELECT password FROM pg_shadow --", "\"; UPDATE transactions SET \"Description\"='pwned'; --",
               "%' AND 1=1 --", "heart'); INSERT INTO transactions VALUES (1); --", "../../etc/passwd"]
    cur.execute("SELECT COUNT(*) AS n FROM transactions;")
    n_before = cur.fetchone()["n"]
    for i, atk in enumerate(ATTACKS, 1):
        safe_by = {}
        for name in BASE_METHODS:
            try:
                run_method(name, atk, None)
                err = False
            except Exception as e:             # noqa: BLE001
                err = True
                print(f"   ⚠️ {name} loi voi chuoi: {atk!r} -> {type(e).__name__}")
            cur.execute("SELECT COUNT(*) AS n FROM transactions;")
            safe = (not err) and cur.fetchone()["n"] == n_before
            safe_by[name] = safe
        for h, m in HYBRIDS.items():                    # hybrid chi ghep ket qua cua bm25 va embedding: an toan neu ca hai an toan
            safe_by[h] = safe_by["bm25"] and safe_by[m]
        save_rows([{"CaseID": f"X{i:02d}", "Split": "test", "Type": "Security", "SubType": "injection",
                    "EvalMode": "security", "Model": nm, "InjectionSafe": float(sf)} for nm, sf in safe_by.items()])

# =====================================================
# [9] TONG HOP
# =====================================================
log = pd.read_csv(LOG_CSV, dtype={"CaseID": str})
ans = log[log.EvalMode != "security"]
ans.groupby(["Type", "SubType", "Model"]).mean(numeric_only=True).round(3).reset_index().to_csv(
    SUMMARY_CSV, index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
order = [m for m in ALL_METHODS if m in set(log.Model)]


def ci(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(42)
    m = rng.choice(x, size=(BOOTSTRAP, len(x))).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


print(f"\n=== [A] TY LE DAT (Success) THEO NHOM - 'co ket qua dung trong top-{K_MAIN}' (kem khoang tin cay 95%) ===")
rows_out = []
for typ in ["Simple", "Complex", "Ambiguous", "ALL"]:
    sub = ans if typ == "ALL" else ans[ans.Type == typ]
    line = {"Nhom": typ, "n": sub.CaseID.nunique()}
    for m in order:
        v = sub[sub.Model == m].Success.dropna().values
        if len(v):
            lo, hi = ci(v)
            line[m] = f"{v.mean():.2f} [{lo:.2f}, {hi:.2f}]"
    rows_out.append(line)
print(pd.DataFrame(rows_out).to_string(index=False))

print("\n=== [B] HIT@K VA MRR (chi ca retrieval) ===")
rt = ans[ans.EvalMode == "retrieval"]
print(rt.groupby(["Type", "Model"])[HIT_COLS + ["MRR", "ResponseTime"]].mean().round(3).to_string())

print("\n=== [C] THEO LOAI CON - Success ===")
print(ans.pivot_table(index=["Type", "SubType"], columns="Model", values="Success", aggfunc="mean")[order]
      .round(2).to_string())

print("\n=== [D] CA PHAI TU CHOI: AbstainOK cao = tot; Hallucination = tra loi khi dang le tu choi ===")
print(ans[ans.EvalMode == "abstain"].groupby(["SubType", "Model"])[["AbstainOK", "Hallucination"]]
      .mean().round(3).to_string())
print("   Ty le he thong dua ra cau tra loi voi ca tra loi duoc (Accepted, ca retrieval; cao la tot):")
print(rt.groupby("Model").Accepted.mean().round(3).reindex(order).to_string())

print("\n=== [E] AUROC TU CHOI (1.0 = tach hoan toan; 0.5 = ngau nhien) ===")
for name in [m for m in DIST_METHODS if m in order]:
    d = ans[ans.Model == name]
    neg = d[d.EvalMode == "retrieval"].Top1Distance.dropna().values
    pos = d[d.EvalMode == "abstain"].Top1Distance.dropna().values
    if len(neg) and len(pos):
        u = mannwhitneyu(pos, neg, alternative="greater").statistic
        print(f"  {name:5s} AUROC = {u / (len(pos) * len(neg)):.3f} | TAU dung = {TAU.get(name, float('nan')):.3f}")

if (log.EvalMode == "security").any():
    print("\n=== [F] AN TOAN SQL INJECTION (1.0 = an toan) ===")
    print(log[log.EvalMode == "security"].groupby("Model").InjectionSafe.mean().round(3).to_string())

cur.close()
conn.close()
print(f"\n🎉 Xong. Log: {LOG_CSV} | Tong hop: {SUMMARY_CSV}")