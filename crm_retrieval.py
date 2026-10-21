"""
crm_retrieval.py - Thu vien truy xuat dung chung cho B9 (rule, BM25, MiniLM, BGE, E5, hybrid).

Tat ca phuong phap chay bang cau SQL tham so hoa trong PostgreSQL (khong noi chuoi nguoi dung vao SQL).
  rule   : ILIKE '%cum tim%' tren Description (khong xep hang)
  bm25   : BM25 Okapi (k1=1.5, b=0.75) tren cac bang chi so bm25_* (xem BM25_SETUP), tai lieu = mo ta san pham duy nhat
  mini/bge/e5 : khoang cach cosine pgvector tren cot embedding1/2/3
  hyb_<m>: ket hop RRF (reciprocal rank fusion, k=60) giua BM25 va embedding m. Hai truy van van chay trong
           PostgreSQL; buoc hop nhat xep hang (cong 1/(k+hang)) lam o ung dung. Thoi gian = tong hai truy van.
Khoang cach quy ve [0,1]: BM25 d = 1/(1+diem); hybrid d = min(d_bm25, d_embedding) cua mo ta xep hang 1 (thieu = 1.0).
"""
import os
import re
import time
from collections import OrderedDict

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

DB = dict(dbname=os.getenv("PGDATABASE", "retaildb"), user=os.getenv("PGUSER", "postgres"),
          password=os.getenv("PGPASSWORD", "postgres"), host=os.getenv("PGHOST", "localhost"),
          port=os.getenv("PGPORT", "5432"))
FETCH = 60                 # so dong lay ve cho moi truy van
TOP_INV = 10               # so hoa don giu lai sau khi loai trung
K_LIST = [1, 3, 5, 10]
K_MAIN = 5
RRF_K = 60
BM25_K1, BM25_B = 1.5, 0.75
EMB_SPEC = {"mini": ("sentence-transformers/all-MiniLM-L6-v2", "embedding1", ""),
            "bge": ("BAAI/bge-large-en-v1.5", "embedding2", ""),
            "e5": ("intfloat/e5-large-v2", "embedding3", "")}
BASE_METHODS = ["rule", "bm25"] + list(EMB_SPEC)
HYBRIDS = {f"hyb_{m}": m for m in EMB_SPEC}          # hyb_mini -> mini ...
ALL_METHODS = BASE_METHODS + list(HYBRIDS)

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN.findall(text.lower())


def clean(text):
    return " ".join(text.split())


# ---------------------------------------------------------------- BM25 index tables (PostgreSQL)
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
    "CREATE TABLE bm25_stats AS SELECT COUNT(*)::float8 AS n_docs, AVG(dl)::float8 AS avgdl FROM bm25_docs;",
    """CREATE TABLE bm25_idf AS
       SELECT t.term, COUNT(*)::int AS df,
              LN(1 + (s.n_docs - COUNT(*) + 0.5) / (COUNT(*) + 0.5))::float8 AS idf
       FROM bm25_terms t, bm25_stats s GROUP BY t.term, s.n_docs;""",
    "ALTER TABLE bm25_idf ADD PRIMARY KEY (term);",
    'CREATE INDEX IF NOT EXISTS transactions_desc_idx ON transactions ("Description");',
    "ANALYZE bm25_docs; ANALYZE bm25_terms; ANALYZE bm25_idf;",
]

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


# ---------------------------------------------------------------- ham thuan Python (kiem thu duoc khong can CSDL)
def rrf_fuse(bm_rows, emb_rows, k=RRF_K, limit=FETCH):
    """Hop nhat hai danh sach dong (da xep theo do lien quan) bang RRF o muc MO TA SAN PHAM.
    Tra ve danh sach dong: InvoiceNo, Description, distance (=min(d_bm25,d_emb)), bm, emb, rrf."""
    def ranks(rows):
        r = OrderedDict()
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
    score = {d: (1.0 / (k + rb[d]) if d in rb else 0.0) + (1.0 / (k + re_[d]) if d in re_ else 0.0)
             for d in set(rb) | set(re_)}
    order = sorted(score, key=lambda d: (-score[d], min(rb.get(d, 10 ** 6), re_.get(d, 10 ** 6)), d))
    out = []
    for d in order:
        b, e = dist_b.get(d, 1.0), dist_e.get(d, 1.0)
        for i in inv[d]:
            out.append({"InvoiceNo": i, "Description": d, "distance": min(b, e), "bm": b, "emb": e, "rrf": score[d]})
        if len(out) >= limit:
            break
    return out[:limit]


def ranked_invoices(rows, n=TOP_INV):
    return list(dict.fromkeys(str(r["InvoiceNo"]) for r in rows))[:n]


def top_distances(rows):
    """(d1, d2): khoang cach cua mo ta hang 1 va mo ta KHAC hang ke tiep (thieu -> 1.0). rule: (nan, nan)."""
    if not rows or rows[0].get("distance") is None:
        return (np.nan, np.nan)
    d1 = float(rows[0]["distance"])
    first = rows[0]["Description"]
    d2 = next((float(r["distance"]) for r in rows[1:] if r["Description"] != first), 1.0)
    return d1, d2


def score_ranking(ranked, truth):
    out = {f"Hit@{k}": float(any(x in truth for x in ranked[:k])) for k in K_LIST}
    first = next((i for i, x in enumerate(ranked) if x in truth), None)
    out["MRR"] = 1 / (first + 1) if first is not None else 0.0
    return out


# ---------------------------------------------------------------- lop truy xuat
class Retriever:
    def __init__(self, load_models=True, rebuild_bm25=False, need_bm25=True):
        self.conn = psycopg2.connect(**DB)
        self.conn.autocommit = True
        self.cur = self.conn.cursor(cursor_factory=RealDictCursor)
        self.models = {}
        if load_models:
            from sentence_transformers import SentenceTransformer
            for k, (ckpt, col, pre) in EMB_SPEC.items():
                self.models[k] = (SentenceTransformer(ckpt), col, pre)
        if need_bm25:
            self.ensure_bm25(rebuild_bm25)

    def ensure_bm25(self, rebuild=False):
        self.cur.execute("SELECT to_regclass('bm25_idf') AS t;")
        if self.cur.fetchone()["t"] is not None and not rebuild:
            print("✅ Dung lai chi so BM25 da co")
            return
        print(">>> Dung chi so BM25 trong PostgreSQL (1 lan)...")
        t0 = time.perf_counter()
        for q in BM25_SETUP:
            self.cur.execute(q)
        print(f"✅ Xong sau {time.perf_counter() - t0:.1f}s")

    # --- tung phuong phap co ban: tra ve (rows, t_enc, t_db)
    def rule(self, text, cust):
        where = 'AND "CustomerID"::text = %s' if cust else ""
        params = [f"%{text}%"] + ([cust] if cust else []) + [FETCH]
        t0 = time.perf_counter()
        self.cur.execute(f"""SELECT "InvoiceNo","Description" FROM transactions
                             WHERE "Description" ILIKE %s {where} LIMIT %s;""", params)
        rows = self.cur.fetchall()
        return rows, 0.0, time.perf_counter() - t0

    def bm25(self, text, cust):
        t0 = time.perf_counter()
        terms = sorted(set(tokenize(text)))
        t_enc = time.perf_counter() - t0
        where = 'WHERE tr."CustomerID"::text = %s' if cust else ""
        params = [terms] + ([cust] if cust else []) + [FETCH]
        t1 = time.perf_counter()
        self.cur.execute(BM25_SQL.replace("{where}", where), params)
        rows = self.cur.fetchall()
        return rows, t_enc, time.perf_counter() - t1

    def semantic(self, text, cust, name):
        model, col, prefix = self.models[name]
        t0 = time.perf_counter()
        q = model.encode(prefix + text, normalize_embeddings=True).tolist()
        t_enc = time.perf_counter() - t0
        where = 'WHERE "CustomerID"::text = %s' if cust else ""
        params = [q] + ([cust] if cust else []) + [q, FETCH]
        t1 = time.perf_counter()
        self.cur.execute(f"""SELECT "InvoiceNo","Description", {col} <=> %s::vector AS distance
                             FROM transactions {where}
                             ORDER BY {col} <=> %s::vector LIMIT %s;""", params)
        rows = self.cur.fetchall()
        return rows, t_enc, time.perf_counter() - t1

    def run_base(self, name, text, cust):
        text = clean(text)
        if name == "rule":
            return self.rule(text, cust)
        if name == "bm25":
            return self.bm25(text, cust)
        return self.semantic(text, cust, name)

    def run_all(self, text, cust, methods=BASE_METHODS):
        """Chay cac phuong phap co ban + hybrid (suy ra tu ket qua da lay). Tra ve {ten: (rows, t_enc, t_db)}."""
        res = {m: self.run_base(m, text, cust) for m in methods if m in BASE_METHODS}
        if "bm25" in res:
            for h, m in HYBRIDS.items():
                if m in res:
                    rows = rrf_fuse(res["bm25"][0], res[m][0])
                    res[h] = (rows, res[m][1], res["bm25"][2] + res[m][2] + res["bm25"][1])
        return res

    def close(self):
        self.cur.close()
        self.conn.close()
