"""
B9.4 - Kiem thu BAO MAT thuc te tren PostgreSQL (thay cho phan chi mo ta RBAC/ma hoa o ban truoc).

  S1  Tiem SQL (injection): ~22 chuoi tan cong x tat ca phuong phap (rule, bm25, mini, bge, e5, hybrid), o ca truong
      cum tim lan truong ma khach hang. An toan = khong doi so dong bang transactions/bm25_docs, bang van ton tai,
      khong chay lenh tre (pg_sleep), khong doi bien phien app.customer_id, truong ma khach khong tra ve dong nao.
  S2  Co lap khach hang bang ROW LEVEL SECURITY (RLS) cua PostgreSQL: gia lap LOI UNG DUNG (bo mat bo loc CustomerID) hoac
      ke tan cong; do so truy van ro ri du lieu cua khach khac trong 4 dieu kien:
        (a) khong RLS, bo mat bo loc          -> doi chung (du kien ro ri)
        (b) RLS bat, bo mat bo loc            -> du kien 0 ro ri
        (c) RLS bat, khong dat khach hang     -> du kien 0 dong (dong an toan khi thieu ngu canh)
        (d) RLS bat, bo loc = khach khac      -> du kien 0 dong (ke tan cong khai bao ma khac)
  S3  Chi phi (overhead) cua RLS (do tre co/khong RLS, cung truy van) va cua nhat ky kiem toan (audit_log).
  S4  Kiem tra TLS cua ket noi hien tai (chi la kiem tra cau hinh, KHONG chung minh bao mat).
Yeu cau: ket noi bang tai khoan SIEU NGUOI DUNG (de tao vai tro/chinh sach). Dau ra: data/security_B9.csv + in tong hop.
Ghi chu: ma hoa AES-256 luc nghi la tinh nang cua tang luu tru (o dia/ban sao luu), khong kiem thu o day.
"""
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crm_retrieval as R

OUT_CSV = "data/security_B9.csv"
N_LEAK = 30            # so cap (nan nhan A, san pham cua khach B) cho moi phuong phap trong S2
N_OVERHEAD = 20        # so truy van do overhead moi phuong phap
SLEEP_S, SLEEP_LIMIT = 30, 25
CLEANUP_RLS = True     # tat RLS va xoa chinh sach sau khi xong (khong anh huong cac thi nghiem khac)
SEED = 42
METHODS_SEC = R.BASE_METHODS            # rule, bm25, mini, bge, e5 (hybrid chi la ghep ket qua, da bao phu)

ATTACKS = ["'; DROP TABLE transactions; --", "x' OR '1'='1", "1; DELETE FROM transactions WHERE 1=1; --",
           "' UNION SELECT password FROM pg_shadow --", "\"; UPDATE transactions SET \"Description\"='pwned'; --",
           "%' AND 1=1 --", "heart'); INSERT INTO transactions VALUES (1); --", "../../etc/passwd",
           "'; TRUNCATE TABLE transactions; --", "'; DROP TABLE bm25_terms; --", "x'; ALTER TABLE transactions DROP COLUMN \"CustomerID\"; --",
           f"x'; SELECT pg_sleep({SLEEP_S}); --", f"' OR (SELECT 1 FROM pg_sleep({SLEEP_S})) IS NOT NULL --",
           "'; SELECT set_config('app.customer_id','99999',false); --", "' OR \"CustomerID\" IS NOT NULL --",
           "\\'; DROP TABLE transactions; --", "’; DROP TABLE transactions; --", "x' /*!50000 OR*/ '1'='1", "{}'; DROP TABLE x;--",
           "A" * 5000, "%" * 200, "heart\x00drop"]


def wilson_upper(k, n, z=1.96):
    if n == 0:
        return np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c + h


def scalar(cur, sql, params=None):
    cur.execute(sql, params)
    r = cur.fetchone()
    return next(iter(r.values())) if r else None


def snapshot(cur):
    return dict(n_tr=scalar(cur, "SELECT COUNT(*) FROM transactions;"), n_docs=scalar(cur, "SELECT COUNT(*) FROM bm25_docs;"),
                n_terms=scalar(cur, "SELECT COUNT(*) FROM bm25_terms;"),
                cols=scalar(cur, "SELECT COUNT(*) FROM information_schema.columns WHERE table_name='transactions';"))


def setup(cur):
    cur.execute("DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='crm_app') THEN CREATE ROLE crm_app NOLOGIN; END IF; END $$;")
    cur.execute("GRANT SELECT ON transactions, bm25_docs, bm25_terms, bm25_idf, bm25_stats TO crm_app;")
    cur.execute("""CREATE TABLE IF NOT EXISTS audit_log (ts timestamptz DEFAULT now(), customer text, method text,
                   query_hash text, n_rows int, ms real);""")
    cur.execute("GRANT INSERT ON audit_log TO crm_app;")
    cur.execute("DROP POLICY IF EXISTS cust_isolation ON transactions;")
    cur.execute("""CREATE POLICY cust_isolation ON transactions FOR SELECT TO crm_app
                   USING ("CustomerID"::text = current_setting('app.customer_id', true));""")


def enable_rls(cur, on):
    cur.execute(f"ALTER TABLE transactions {'ENABLE' if on else 'DISABLE'} ROW LEVEL SECURITY;")


def set_ctx(cur, cust):
    cur.execute("SELECT set_config('app.customer_id', %s, false);", (cust or "",))


def run(ret, method, text, cust):
    rows, _, _ = ret.run_base(method, text, cust)
    return rows


def main():
    rng = np.random.default_rng(SEED)
    ret = R.Retriever(load_models=True, need_bm25=True)
    cur = ret.cur
    results = []

    def log(test, method, **kw):
        results.append(dict(Test=test, Method=method, **kw))

    # ---------------------------------------------------------------- S4 TLS
    cur.execute("SELECT ssl, version, cipher FROM pg_stat_ssl WHERE pid = pg_backend_pid();")
    ssl = cur.fetchone()
    print(f"[S4] Ket noi dang dung TLS: {ssl['ssl']} ({ssl['version']}, {ssl['cipher']})  (cau hinh phia may chu: SHOW ssl = {scalar(cur, 'SHOW ssl;')})")
    log("S4_tls", "all", n=1, value=float(bool(ssl["ssl"])))

    setup(cur)
    enable_rls(cur, False)

    # ---------------------------------------------------------------- S1 injection
    print("\n[S1] Tiem SQL...")
    base = snapshot(cur)
    set_ctx(cur, "")
    ctx_before = scalar(cur, "SELECT current_setting('app.customer_id', true);")
    for k, atk in enumerate(ATTACKS, 1):
        for m in METHODS_SEC:
            for field in ("text", "customer"):
                err, nrows, t0 = None, 0, time.perf_counter()
                try:
                    rows = run(ret, m, atk if field == "text" else "heart", None if field == "text" else atk)
                    nrows = len(rows)
                except Exception as e:                      # noqa: BLE001
                    err = type(e).__name__
                el = time.perf_counter() - t0
                after = snapshot(cur)
                ctx_after = scalar(cur, "SELECT current_setting('app.customer_id', true);")
                side_effect = after != base or ctx_after != ctx_before
                slow = el > SLEEP_LIMIT
                leak = field == "customer" and nrows > 0 and not atk.startswith("%")
                log("S1_injection", m, payload=k, field=field, error=err or "", side_effect=float(side_effect),
                    slow=float(slow), leak=float(leak), safe=float(not (side_effect or slow or leak)), no_error=float(err is None))
    s1 = pd.DataFrame([r for r in results if r["Test"] == "S1_injection"])
    n = len(s1)
    print(f"   {n} luot thu ({len(ATTACKS)} chuoi x {len(METHODS_SEC)} phuong phap x 2 truong). "
          f"An toan (khong tac dung phu): {int(s1.safe.sum())}/{n}; khong loi ngoai le: {int(s1.no_error.sum())}/{n}; "
          f"can tren 95% cua ty le that bai: {wilson_upper(n - int(s1.safe.sum()), n):.3f}")
    if (s1.no_error == 0).any():
        print("   Cac truong hop co ngoai le (van an toan neu safe=1):", s1[s1.no_error == 0].groupby("error").size().to_dict())

    # ---------------------------------------------------------------- S2 co lap khach hang
    print("\n[S2] Co lap khach hang (RLS)...")
    cur.execute('SELECT DISTINCT "InvoiceNo"::text AS inv, "CustomerID"::text AS cust, "Description" AS dsc '
                'FROM transactions WHERE "CustomerID" IS NOT NULL AND "Description" IS NOT NULL;')
    pairs = pd.DataFrame(cur.fetchall())
    owner = pairs.drop_duplicates("inv").set_index("inv").cust.to_dict()
    by_cust = pairs.groupby("cust").dsc.agg(set).to_dict()
    custs = list(by_cust)
    cases = []
    while len(cases) < N_LEAK:
        b = custs[rng.integers(len(custs))]
        d = sorted(by_cust[b])[rng.integers(len(by_cust[b]))]
        a = custs[rng.integers(len(custs))]
        if a != b and d not in by_cust[a] and "%" not in d and "_" not in d:
            cases.append((a, b, d))
    conds = [("a_noRLS_filter_dropped", False, lambda a, b: (a, None)),
             ("b_RLS_filter_dropped", True, lambda a, b: (a, None)),
             ("c_RLS_no_context", True, lambda a, b: ("", None)),
             ("d_RLS_attacker_filter", True, lambda a, b: (a, b))]
    for cname, rls, f in conds:
        for m in METHODS_SEC:
            leaked_q = foreign_rows = total_rows = 0
            for (a, b, d) in cases:
                ctx, filt = f(a, b)
                enable_rls(cur, rls)
                if rls:
                    cur.execute("SET ROLE crm_app;")
                    set_ctx(cur, ctx)
                try:
                    rows = run(ret, m, d, filt)
                finally:
                    if rls:
                        cur.execute("RESET ROLE;")
                total_rows += len(rows)
                fr = sum(1 for r in rows if owner.get(str(r["InvoiceNo"])) != a)
                foreign_rows += fr
                leaked_q += fr > 0
            enable_rls(cur, False)
            log("S2_isolation", m, condition=cname, n=len(cases), leaked_queries=leaked_q, foreign_rows=foreign_rows, total_rows=total_rows)
            print(f"   {cname:26s} {m:5s}: truy van ro ri {leaked_q}/{len(cases)}, dong cua khach khac {foreign_rows}/{total_rows}")

    # ---------------------------------------------------------------- S3 overhead
    print("\n[S3] Chi phi RLS va nhat ky kiem toan...")
    sample = cases[:N_OVERHEAD]
    for m in METHODS_SEC:
        t_off, t_on = [], []
        for (a, b, d) in sample:
            enable_rls(cur, False)
            t0 = time.perf_counter(); run(ret, m, d, a); t_off.append(time.perf_counter() - t0)
            enable_rls(cur, True)
            cur.execute("SET ROLE crm_app;"); set_ctx(cur, a)
            t0 = time.perf_counter(); run(ret, m, d, a); t_on.append(time.perf_counter() - t0)
            cur.execute("RESET ROLE;")
        enable_rls(cur, False)
        log("S3_rls_overhead", m, n=len(sample), median_off_s=float(np.median(t_off)), median_on_s=float(np.median(t_on)),
            ratio=float(np.median(t_on) / np.median(t_off)))
        print(f"   RLS {m:5s}: trung vi {np.median(t_off):.3f}s -> {np.median(t_on):.3f}s (x{np.median(t_on) / np.median(t_off):.2f})")
    cur.execute("TRUNCATE audit_log;")
    ins = []
    for (a, b, d) in sample:
        t0 = time.perf_counter()
        cur.execute("INSERT INTO audit_log (customer, method, query_hash, n_rows, ms) VALUES (%s,%s,%s,%s,%s);",
                    (a, "bm25", hashlib.sha256(d.encode()).hexdigest()[:16], 0, 0.0))
        ins.append(time.perf_counter() - t0)
    n_log = scalar(cur, "SELECT COUNT(*) FROM audit_log;")
    log("S3_audit", "all", n=len(sample), logged=int(n_log), median_insert_ms=float(np.median(ins) * 1000))
    print(f"   Nhat ky kiem toan: ghi {n_log}/{len(sample)} su kien, trung vi {np.median(ins) * 1000:.2f} ms moi lan ghi")
    cur.execute("TRUNCATE audit_log;")

    if CLEANUP_RLS:
        enable_rls(cur, False)
        cur.execute("DROP POLICY IF EXISTS cust_isolation ON transactions;")
    ret.close()
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n✅ Da luu {OUT_CSV}")


if __name__ == "__main__":
    main()
