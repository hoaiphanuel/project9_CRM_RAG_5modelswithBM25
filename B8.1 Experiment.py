"""
B8.1 (v2) - Chay thu nghiem 4 phuong phap (rule-based, MiniLM, BGE, E5) tren bo test B8.0.

Cach chay giong ban cu: tung ca mot, in ket qua tung phuong phap, tinh diem tich luy,
ghi log ngay sau moi ca (dung giua chung van con ket qua).

Cach cham:
  * Ca retrieval : THANH CONG neu top-K co it nhat 1 hoa don thuoc tap dap an (Hit@K).
                   Bao cao Hit@1/3/5/10 va MRR de thay muc do xep hang.
  * Ca abstain   : THANH CONG neu he thong tu choi (rule: khong co dong nao;
                   embedding: khoang cach top-1 > nguong TAU).
  * Nguong TAU moi mo hinh duoc CHON TU DONG tren tap dev (Split = dev), roi moi ap dung len tap test.
    Khong chinh nguong tren chinh tap test.
Dau ra: data/test_results_B7.csv, data/summary_B7.csv
"""
import os
import re
import time

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
print(">>> [3] Nap mo hinh embedding...")
MODELS = {
    "mini": (SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"), "embedding1", ""),
    "bge":  (SentenceTransformer("BAAI/bge-large-en-v1.5"), "embedding2", ""),
    "e5":   (SentenceTransformer("intfloat/e5-large-v2"), "embedding3", ""),
}
METHODS = ["rule"] + list(MODELS)
print("✅ Da nap: " + ", ".join(MODELS))


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
    return rule_based(text, cust) if name == "rule" else semantic(text, cust, name)


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
for m in METHODS:
    run_method(m, "warm up", None)

TAU = {m: TAU_DEFAULT for m in MODELS}
if AUTO_TAU and len(dev):
    print(">>> Chon nguong tu choi TAU tren tap dev (khong dung tap test)...")
    for name in MODELS:
        d_ans, d_abs = [], []
        for _, c in dev.iterrows():
            rows, _, _ = run_method(name, c.SearchText, c.CustomerID or None)
            if rows:
                (d_abs if c.EvalMode == "abstain" else d_ans).append(float(rows[0]["distance"]))
        d_ans, d_abs = np.array(d_ans), np.array(d_abs)
        if len(d_ans) and len(d_abs):
            allv = np.unique(np.concatenate([d_ans, d_abs]))
            cands = np.concatenate([[allv[0] - 1e-6], (allv[:-1] + allv[1:]) / 2, [allv[-1] + 1e-6]])
            best = max(cands, key=lambda t: np.mean(d_ans <= t) - np.mean(d_abs <= t))   # Youden J
            TAU[name] = float(best)
            print(f"   {name:5s} TAU={best:.3f} | giu lai ca tra loi duoc: {np.mean(d_ans <= best):.0%} | "
                  f"nhan nham ca phai tu choi: {np.mean(d_abs <= best):.0%}")
else:
    print(f"   Dung TAU mac dinh = {TAU_DEFAULT}")

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
        for name in METHODS:
            rows, t_enc, t_db = run_method(name, c.SearchText, cust)
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

            if c.EvalMode == "abstain":
                row["AbstainOK"] = float(not ok_answer)
                row["Hallucination"] = float(ok_answer)       # tra loi khi dang le phai tu choi
                row["Success"] = row["AbstainOK"]
                run_abs[name].append(row["AbstainOK"])
                print(f"[{name.upper():4s}] {'DAT ' if row['Success'] else 'SAI '} AbstainOK={bool(row['AbstainOK'])}"
                      f"{dist_txt}, Time={row['ResponseTime']:.3f}s")
            else:
                row.update(score(ranked_invoices(rows), truth))
                print(f"[{name.upper():4s}] {'DAT ' if row['Success'] else 'SAI '} Hit@1={bool(row['Hit@1'])}, "
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
        for name in METHODS:
            try:
                run_method(name, atk, None)
                err = False
            except Exception as e:             # noqa: BLE001
                err = True
                print(f"   ⚠️ {name} loi voi chuoi: {atk!r} -> {type(e).__name__}")
            cur.execute("SELECT COUNT(*) AS n FROM transactions;")
            safe = (not err) and cur.fetchone()["n"] == n_before
            save_rows([{"CaseID": f"X{i:02d}", "Split": "test", "Type": "Security", "SubType": "injection",
                        "EvalMode": "security", "Model": name, "InjectionSafe": float(safe)}])

# =====================================================
# [9] TONG HOP
# =====================================================
log = pd.read_csv(LOG_CSV, dtype={"CaseID": str})
ans = log[log.EvalMode != "security"]
ans.groupby(["Type", "SubType", "Model"]).mean(numeric_only=True).round(3).reset_index().to_csv(
    SUMMARY_CSV, index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
order = METHODS


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
for name in MODELS:
    d = ans[ans.Model == name]
    neg = d[d.EvalMode == "retrieval"].Top1Distance.dropna().values
    pos = d[d.EvalMode == "abstain"].Top1Distance.dropna().values
    if len(neg) and len(pos):
        u = mannwhitneyu(pos, neg, alternative="greater").statistic
        print(f"  {name:5s} AUROC = {u / (len(pos) * len(neg)):.3f} | TAU dung = {TAU[name]:.3f}")

if (log.EvalMode == "security").any():
    print("\n=== [F] AN TOAN SQL INJECTION (1.0 = an toan) ===")
    print(log[log.EvalMode == "security"].groupby("Model").InjectionSafe.mean().round(3).to_string())

cur.close()
conn.close()
print(f"\n🎉 Xong. Log: {LOG_CSV} | Tong hop: {SUMMARY_CSV}")