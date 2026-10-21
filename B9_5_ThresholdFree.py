"""
B9.5 - Phan tich KHONG PHU THUOC cach chon nguong (offline, doc data/results_B9.csv) + chinh sach "kiem soat ty le tra loi nham".

Muc dich: sua van de cua B9_2 (chinh sach toi da hoa do chinh xac o ty le tu choi 20% lam embedding thanh "tra loi tat ca").
(A) Chi so khong can nguong, tinh tren tap TEST:
      AUROC  : kha nang tach "tra loi se dung" (truy xuat trung Hit@5) khoi "phai tu choi"
      AURC   : dien tich duoi duong rui ro-do phu (thap hon la tot)
      Tra loi duoc a% truy van hop le khi chap nhan tra loi nham FA% ca phai tu choi (FA = 5, 10, 20)  [chan doan: nguong lay tu chinh tap test]
(B) Chinh sach P4 "kiem soat tra loi nham" (HOC TREN DEV, ap len test mot lan):
      tau = phan vi alpha cua khoang cach top-1 cac ca phai tu choi trong dev  (alpha = 5%, 10%)
      => ky vong chi alpha ca phai tu choi bi tra loi nham. DAY LA PHAN TICH BO SUNG SAU KHI THAY KET QUA B9_2, ghi ro trong bai.
Pham vi bao cao: core (500 ca B8) ; core+hard_negative ; moi nhom bo sung rieng le.
Dau ra: data/threshold_free_B9.csv, data/p4_policy_B9.csv, data/risk_coverage_free_B9.png
"""
import numpy as np
import pandas as pd
from scipy import stats

LOG_CSV = "data/results_B9.csv"
OUT = "data"
METHODS = ["bm25", "mini", "bge", "e5", "hyb_mini", "hyb_bge", "hyb_e5"]
LABEL = {"rule": "Rule-based", "bm25": "BM25", "mini": "MiniLM", "bge": "BGE", "e5": "E5",
         "hyb_mini": "BM25+MiniLM", "hyb_bge": "BM25+BGE", "hyb_e5": "BM25+E5"}
FA_LEVELS = [0.05, 0.10, 0.20]
ALPHAS = [0.05, 0.10]
B = 3000
rng = np.random.default_rng(42)


def prep(d):
    y = (d.EvalMode == "retrieval").values
    hit = np.where(y, d["Hit@5"].fillna(0).values, 0) == 1
    return d.Top1Distance.fillna(1.0).values.astype(float), y, hit, d.CaseID.values


def auroc(dist, y, hit):
    pos, neg = dist[y & hit], dist[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return stats.mannwhitneyu(neg, pos, alternative="greater").statistic / (len(pos) * len(neg))


def aurc(dist, y, hit):
    err = ~(y & hit)
    o = np.argsort(dist, kind="stable")
    risk = np.cumsum(err[o]) / np.arange(1, len(err) + 1)
    return float(risk.mean())


def answered_at_fa(dist, y, hit, fa):
    pos, neg = dist[y & hit], dist[~y]
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    return float((pos <= np.quantile(neg, fa)).mean())


def boot(fn, dist, y, hit, n=B):
    idx = np.arange(len(dist))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        v = fn(dist[s], y[s], hit[s])
        if not np.isnan(v):
            vals.append(v)
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def main():
    log = pd.read_csv(LOG_CSV, dtype={"CaseID": str})
    if "Module" not in log.columns:
        log["Module"] = "core"
    test = log[(log.Split == "test") & (~log.Module.str.endswith("_Q"))]
    dev = log[(log.Split == "dev") & (~log.Module.str.endswith("_Q"))]
    scopes = {"core": ["core"], "core+hard_negative": ["core", "hard_negative"]}
    rows = []
    for sc, mods in scopes.items():
        for m in METHODS:
            d = test[(test.Model == m) & test.Module.isin(mods)].sort_values("CaseID")
            dist, y, hit, _ = prep(d)
            r = dict(scope=sc, Method=LABEL[m], n=len(d), n_out_of_scope=int((~y).sum()))
            r["AUROC"] = auroc(dist, y, hit)
            r["AUROC_lo"], r["AUROC_hi"] = boot(auroc, dist, y, hit, 1000)
            r["AURC"] = aurc(dist, y, hit)
            for fa in FA_LEVELS:
                r[f"answered@FA{int(fa * 100)}"] = answered_at_fa(dist, y, hit, fa)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/threshold_free_B9.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    for sc in scopes:
        print(f"\\n=== (A) KHONG PHU THUOC NGUONG - {sc} (tap test) ===")
        print(df[df.scope == sc].drop(columns="scope").round(3).to_string(index=False))

    # ---- (B) P4
    out = []
    for dev_scope, dev_mods in (("dev_core", ["core"]), ("dev_all", None)):
        for m in METHODS:
            dd = dev[(dev.Model == m) & ((dev.Module.isin(dev_mods)) if dev_mods else True)]
            dist_d, y_d, _, _ = prep(dd.sort_values("CaseID"))
            neg_d = dist_d[~y_d]
            for a in ALPHAS:
                tau = float(np.quantile(neg_d, a)) if len(neg_d) else np.nan
                for sc, mods in scopes.items():
                    d = test[(test.Model == m) & test.Module.isin(mods)].sort_values("CaseID")
                    dist, y, hit, _ = prep(d)
                    acc = dist <= tau
                    cor = np.where(y, acc & hit, ~acc)
                    out.append(dict(dev=dev_scope, alpha=a, scope=sc, Method=LABEL[m], tau=tau,
                                    e2e=cor.mean(), answer_rate=acc[y].mean(), refusal=(~acc[~y]).mean(),
                                    hallucinated=int(acc[~y].sum()), n_out=int((~y).sum()), false_refusals=int((~acc[y]).sum())))
    p4 = pd.DataFrame(out)
    p4.to_csv(f"{OUT}/p4_policy_B9.csv", index=False)
    for dsc in ("dev_core", "dev_all"):
        for sc in scopes:
            print(f"\\n=== (B) P4 kiem soat tra loi nham: dev={dsc}, test={sc} ===")
            t = p4[(p4.dev == dsc) & (p4.scope == sc)]
            print(t.pivot_table(index="Method", columns="alpha", values=["e2e", "answer_rate", "refusal"]).round(3).to_string())

    # ---- ve duong rui ro - do phu (core+hard_negative)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
    for k, (sc, mods) in enumerate(scopes.items()):
        for m in METHODS:
            d = test[(test.Model == m) & test.Module.isin(mods)].sort_values("CaseID")
            dist, y, hit, _ = prep(d)
            err = ~(y & hit)
            o = np.argsort(dist, kind="stable")
            risk = np.cumsum(err[o]) / np.arange(1, len(err) + 1)
            ax[k].plot(np.arange(1, len(err) + 1) / len(err), risk, label=LABEL[m], lw=1.3)
        ax[k].set_title(f"Test: {sc}")
        ax[k].set_xlabel("Coverage (share of queries answered)")
        ax[k].grid(alpha=.3)
    ax[0].set_ylabel("Risk (error rate among answered)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/risk_coverage_free_B9.png", dpi=300)
    print("\\n✅ Da luu threshold_free_B9.csv, p4_policy_B9.csv, risk_coverage_free_B9.png")


if __name__ == "__main__":
    main()
