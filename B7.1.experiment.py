"""
B7.1 - Chạy thí nghiệm: Rule-based (ILIKE) vs MiniLM vs BGE vs E5 trên PostgreSQL + pgvector.

Điểm khác so với bản cũ (xem chi tiết trong phần giải thích):
  * Dùng SearchText + bộ lọc (QtyGt / InvoiceTotalGt / DateFrom / DateTo) của test case
    -> áp dụng GIỐNG HỆT nhau cho cả 4 phương pháp (so sánh công bằng phần tìm kiếm).
  * Ground truth là tập sản phẩm; đo Hit@1, Hit@5, Recall@5, MRR, Precision@5.
  * Có ngưỡng tương đồng (tau) để hệ thống "từ chối trả lời" -> đo hallucination / false-positive.
  * Cosine distance (<=>) + vector chuẩn hoá; thêm prefix "query: " cho E5.
  * Tách thời gian encode / DB; warm-up; try/except + autocommit (1 lỗi không làm hỏng cả lượt chạy).
  * Kiểm tra bảo mật thực sự: rò rỉ dữ liệu khách khác, bảng bị thay đổi, lỗi không xử lý.
  * Chống wildcard LIKE (% và _) bằng ESCAPE.
"""
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

# ============================ CẤU HÌNH ============================
DATA_DIR = os.getenv("B7_DATA_DIR", "data")
DB = dict(dbname=os.getenv("PGDATABASE", "retaildb"), user=os.getenv("PGUSER", "postgres"),
          password=os.getenv("PGPASSWORD", "postgres"), host=os.getenv("PGHOST", "localhost"),
          port=os.getenv("PGPORT", "5432"))
TABLE = "transactions"
TOP_K = 5
ESCAPE_LIKE = True            # False = hành vi cũ (% và _ trong chuỗi tìm kiếm là wildcard)

# name -> (HuggingFace id, cột vector, prefix cho query, tau mặc định)
# tau = ngưỡng cosine similarity tối thiểu để coi là "có câu trả lời". Đây chỉ là giá trị khởi điểm:
# hãy hiệu chỉnh lại bằng B7.2 (--calibrate) rồi chạy lại phần phân tích.
MODELS = {
    "mini": ("sentence-transformers/all-MiniLM-L6-v2", "embedding1", "", 0.60),
    "bge":  ("BAAI/bge-large-en-v1.5",                 "embedding2", "", 0.75),
    "e5":   ("intfloat/e5-large-v2",                   "embedding3", "query: ", 0.80),
}


def norm(s):
    return str(s).strip().lower() if s is not None else ""


# ============================ [1] TEST CASES ============================
print(">>> [1] Loading test cases...")
raw = pd.read_csv(os.path.join(DATA_DIR, "test_cases_B7.csv"), dtype=str, keep_default_na=False)


def opt_num(x, cast):
    return cast(x) if x != "" else None


cases = []
for _, r in raw.iterrows():
    cases.append({
        **r.to_dict(),
        "gt": [norm(x) for x in json.loads(r["ExpectedDescriptions"])],
        "qty_gt": opt_num(r["QtyGt"], float), "inv_gt": opt_num(r["InvoiceTotalGt"], float),
        "date_from": r["DateFrom"] or None, "date_to": r["DateTo"] or None,
        "logic": r["FilterLogic"] if r["FilterLogic"] in ("AND", "OR") else "AND",   # whitelist
        "search": r["SearchText"].replace("<NUL>", "\x00"),
        "should_abstain": r["ShouldAbstain"] == "True",
        "is_injection": r["IsInjection"] == "True",
    })
print(f"✅ {len(cases)} test cases")

# ============================ [2] DATABASE ============================
print(">>> [2] Connecting to PostgreSQL...")
conn = psycopg2.connect(**DB)
conn.autocommit = True                      # lỗi 1 câu lệnh không làm "abort" cả transaction
cur = conn.cursor(cursor_factory=RealDictCursor)
cur.execute(f'SELECT count(*) AS n FROM {TABLE}')
BASELINE_ROWS = cur.fetchone()["n"]
print(f"✅ Connected. {TABLE} has {BASELINE_ROWS:,} rows")

# ============================ [3] MODELS ============================
print(">>> [3] Loading embedding models...")
encoders = {}
for name, (hf_id, col, prefix, tau) in MODELS.items():
    m = SentenceTransformer(hf_id)
    m.encode("warm-up", normalize_embeddings=True)       # warm-up để không tính thời gian khởi tạo
    encoders[name] = m
print("✅ Models loaded:", ", ".join(MODELS))


# ============================ [4] SEARCH ============================
def filter_sql(case, params):
    parts, amt = [], []
    if case["qty_gt"] is not None:
        amt.append('"Quantity" > %(qty_gt)s'); params["qty_gt"] = case["qty_gt"]
    if case["inv_gt"] is not None:
        amt.append(f'"InvoiceNo" IN (SELECT "InvoiceNo" FROM {TABLE} GROUP BY "InvoiceNo" '
                   f'HAVING SUM("Revenue") > %(inv_gt)s)')
        params["inv_gt"] = case["inv_gt"]
    if amt:
        parts.append("(" + f" {case['logic']} ".join(amt) + ")")
    if case["date_from"]:
        parts.append('"InvoiceDate"::timestamp >= %(d_from)s::timestamp'); params["d_from"] = case["date_from"]
    if case["date_to"]:
        parts.append('"InvoiceDate"::timestamp < (%(d_to)s::date + 1)'); params["d_to"] = case["date_to"]
    return "".join(f" AND {p}" for p in parts)


def like_pattern(text):
    t = text.strip()
    if ESCAPE_LIKE:
        t = t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{t}%"


def rule_based(case):
    params = {"cust": case["CustomerID"], "pat": like_pattern(case["search"]), "k": TOP_K}
    f = filter_sql(case, params)
    # 1 dòng / sản phẩm; xếp hạng thô: mô tả ngắn nhất (gần chuỗi tìm kiếm nhất) lên trước
    cur.execute(rf"""
        SELECT * FROM (
          SELECT DISTINCT ON (lower(trim("Description")))
                 "InvoiceNo","StockCode","Description","CustomerID"
          FROM {TABLE}
          WHERE "CustomerID"::text = %(cust)s AND "Description" ILIKE %(pat)s ESCAPE '\' {f}
          ORDER BY lower(trim("Description")), "InvoiceNo"
        ) t ORDER BY length("Description"), "Description" LIMIT %(k)s;""", params)
    return cur.fetchall()


def semantic(case, name):
    hf_id, col, prefix, _ = MODELS[name]
    t0 = time.perf_counter()
    emb = encoders[name].encode(prefix + case["search"], normalize_embeddings=True)
    t_enc = time.perf_counter() - t0
    params = {"cust": case["CustomerID"], "emb": "[" + ",".join(f"{x:.6f}" for x in emb) + "]", "k": TOP_K}
    f = filter_sql(case, params)
    t1 = time.perf_counter()
    cur.execute(f"""
        SELECT * FROM (
          SELECT DISTINCT ON (lower(trim("Description")))
                 "InvoiceNo","StockCode","Description","CustomerID", {col} <=> %(emb)s::vector AS distance
          FROM {TABLE}
          WHERE "CustomerID"::text = %(cust)s AND {col} IS NOT NULL {f}
          ORDER BY lower(trim("Description")), distance
        ) t ORDER BY distance LIMIT %(k)s;""", params)
    rows = cur.fetchall()
    return rows, t_enc, time.perf_counter() - t1


# ============================ [5] METRICS ============================
def token_f1(pred, truth):
    p, t = set(norm(pred).split()), set(norm(truth).split())
    if not p or not t:
        return 0.0
    tp = len(p & t)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(t)
    return 2 * prec * rec / (prec + rec)


def evaluate(rows, case, name):
    semantic_m = name != "rule"
    gt = set(case["gt"])
    descs = [norm(r["Description"]) for r in rows]
    top1 = descs[0] if descs else None
    top1_sim = (1.0 - float(rows[0]["distance"])) if (semantic_m and rows) else None
    tau = MODELS[name][3] if semantic_m else None
    answered = bool(rows) and (not semantic_m or top1_sim >= tau)
    top1_ok = bool(top1 in gt) if gt else False
    out = {"Answered": answered, "Tau": tau, "Top1Sim": top1_sim, "Top1Correct": top1_ok,
           "Top1Desc": top1, "Top5": json.dumps(descs, ensure_ascii=False)}
    if gt:                                           # ca dương: có đáp án
        rel = [i for i, d in enumerate(descs) if d in gt]
        out.update({
            "Hit@1": float(top1_ok),
            "Hit@5": float(bool(rel)),
            "Recall@5": len({descs[i] for i in rel}) / min(len(gt), TOP_K),
            "Precision@5": (len(rel) / len(descs)) if descs else 0.0,
            "MRR": (1.0 / (rel[0] + 1)) if rel else 0.0,
            "ExactMatch": float(top1 in gt),
            "F1": max(token_f1(top1, g) for g in gt) if top1 else 0.0,
            "FalseAbstain": float(not answered),
        })
    else:                                            # ca âm: đáp án đúng là "không có"
        out["AbstainCorrect"] = float(not answered)
    out["Hallucination"] = float(answered and not top1_ok)          # trả lời nhưng sai (hoặc không nên trả lời)
    out["EndToEndCorrect"] = float((answered and top1_ok) if gt else (not answered))
    out["Leak"] = any(str(r["CustomerID"]) != case["CustomerID"] for r in rows)
    return out


# ============================ [6] RUN ============================
print(f">>> [6] Running {len(cases)} cases x {1 + len(MODELS)} methods...")
logs = []
try:
    for i, case in enumerate(cases, 1):
        for name in ["rule"] + list(MODELS):
            rows, err, t_enc, t_db = [], "", 0.0, 0.0
            try:
                if name == "rule":
                    t0 = time.perf_counter(); rows = rule_based(case); t_db = time.perf_counter() - t0
                else:
                    rows, t_enc, t_db = semantic(case, name)
            except Exception as e:                              # lỗi được ghi nhận, không dừng chương trình
                err = f"{type(e).__name__}: {str(e)[:120]}"
                if conn.closed:
                    conn = psycopg2.connect(**DB); conn.autocommit = True
                    cur = conn.cursor(cursor_factory=RealDictCursor)
            res = evaluate(rows, case, name)
            table_changed = False
            if case["Type"] == "Security":                      # bảng còn nguyên không?
                try:
                    cur.execute(f"SELECT count(*) AS n FROM {TABLE}")
                    table_changed = cur.fetchone()["n"] != BASELINE_ROWS
                except Exception:
                    table_changed = True
            violation = bool(res["Leak"] or table_changed)
            logs.append({
                "CaseID": case["CaseID"], "Type": case["Type"], "Subtype": case["Subtype"],
                "Variant": case["Variant"], "CustomerTier": case["CustomerTier"], "Model": name,
                "ShouldAbstain": case["should_abstain"], "IsInjection": case["is_injection"],
                **res, "Error": err, "HasError": bool(err), "TableChanged": table_changed,
                "SecurityViolation": violation,
                "InjectionBlocking": float(not violation and not err) if case["is_injection"] else np.nan,
                "EncodeTime": t_enc, "DbTime": t_db, "ResponseTime": t_enc + t_db,
            })
        if i % 25 == 0 or i == len(cases):
            print(f"   {i}/{len(cases)} done")
finally:
    df_log = pd.DataFrame(logs)
    df_log.to_csv(os.path.join(DATA_DIR, "test_results_B7.csv"), index=False)
    print("🎉 Saved test_results_B7.csv", len(df_log), "rows")

# ============================ [7] SUMMARY ============================
pos = df_log[(~df_log["ShouldAbstain"]) & (df_log["Type"] != "Security")]
neg = df_log[(df_log["ShouldAbstain"]) & (df_log["Type"] != "Security")]
sec = df_log[df_log["Type"] == "Security"]
pd.set_option("display.width", 200)
print("\n=== POSITIVE cases (Simple/Ambiguous/Complex) ===")
print(pos.groupby(["Type", "Model"])[["Hit@1", "Hit@5", "Recall@5", "MRR", "F1", "Hallucination",
                                       "EndToEndCorrect", "ResponseTime"]].mean().round(3))
print("\n=== NEGATIVE cases (đúng = không trả lời) ===")
print(neg.groupby(["Subtype", "Model"])[["AbstainCorrect", "Hallucination"]].mean().round(3).unstack())
print("\n=== SECURITY ===")
print(sec.groupby("Model")[["SecurityViolation", "HasError", "InjectionBlocking"]].mean().round(3))
cur.close(); conn.close()
print("✅ Done")