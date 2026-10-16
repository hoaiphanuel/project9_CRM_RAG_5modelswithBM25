"""
B8.2 - Thong ke so sanh 4 phuong phap (rule, mini, bge, e5) tu file log cua B8.1.

  * Khoang tin cay 95% bang bootstrap (lay mau lai co hoan lai cac ca test)
  * Kiem dinh so sanh CAP tren cung cac ca: McNemar chinh xac cho chi so nhi phan (Success, Hit@k),
    Wilcoxon cho chi so lien tuc (MRR); hieu chinh da so sanh Holm (6 cap)
Dau ra: data/ci_B7.csv, data/pairwise_B7.csv
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

LOG_CSV = "data/test_results_B8.csv"
MODELS = ["rule", "mini", "bge", "e5"]
METRICS = ["Success", "Hit@1", "MRR"]       # chon truoc chi so chinh (Success) de tranh san p-value
B = 10000
rng = np.random.default_rng(42)

log = pd.read_csv(LOG_CSV, dtype={"CaseID": str})
log = log[log.EvalMode != "security"]


def wide(sub, metric):
    w = sub.pivot(index="CaseID", columns="Model", values=metric)
    return w.reindex(columns=MODELS).dropna()


def boot_ci(x):
    m = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def holm(p):
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def paired_p(a, b):
    if np.allclose(a, b):
        return 1.0
    if set(np.unique(a)) | set(np.unique(b)) <= {0.0, 1.0}:
        n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        return stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return stats.wilcoxon(a, b).pvalue


def analyze(sub, label, metrics):
    ci_rows, te_rows = [], []
    for met in metrics:
        w = wide(sub[sub[met].notna()] if met in sub else sub, met) if met in sub else None
        if w is None or len(w) < 5:
            continue
        for m in MODELS:
            lo, hi = boot_ci(w[m].values)
            ci_rows.append({"Nhom": label, "ChiSo": met, "MoHinh": m, "n": len(w),
                            "TB": w[m].mean(), "CI_thap": lo, "CI_cao": hi})
        pairs = list(itertools.combinations(MODELS, 2))
        p = [paired_p(w[a].values, w[b].values) for a, b in pairs]
        pa = holm(np.array(p))
        for (a, b), pv, padj in zip(pairs, p, pa):
            te_rows.append({"Nhom": label, "ChiSo": met, "Cap": f"{a} vs {b}",
                            "Chenh": w[a].mean() - w[b].mean(), "p": pv, "p_Holm": padj, "YNghia": padj < 0.05})
    return ci_rows, te_rows


groups = [("Tat ca", log)] + [(t, log[log.Type == t]) for t in ["Simple", "Complex", "Ambiguous"]]
groups += [(f"{t}/{s}", log[(log.Type == t) & (log.SubType == s)])      # tham do (exploratory), n nho
           for t, s in log[["Type", "SubType"]].drop_duplicates().values]
# Phan tich rieng ca retrieval cho Hit@k va MRR (ca abstain khong co MRR)
ci_all, te_all = [], []
for label, sub in groups:
    c, t = analyze(sub, label, METRICS)
    ci_all += c
    te_all += t

ci = pd.DataFrame(ci_all).round(4)
te = pd.DataFrame(te_all).round(4)
ci.to_csv("data/ci_B7.csv", index=False)
te.to_csv("data/pairwise_B7.csv", index=False)

main = ["Tat ca", "Simple", "Complex", "Ambiguous"]
print("=== TY LE DAT (Success) - TB [95% CI] ===")
s = ci[(ci.ChiSo == "Success") & ci.Nhom.isin(main)].copy()
s["gia_tri"] = s.apply(lambda r: f"{r.TB:.2f} [{r.CI_thap:.2f}, {r.CI_cao:.2f}]", axis=1)
print(s.pivot(index="Nhom", columns="MoHinh", values="gia_tri").reindex(main)[MODELS].to_string())

print("\n=== SO SANH CAP (Success, hieu chinh Holm) ===")
print(te[(te.ChiSo == "Success") & te.Nhom.isin(main)][["Nhom", "Cap", "Chenh", "p", "p_Holm", "YNghia"]]
      .to_string(index=False))
print("\nLuu y: phan theo loai con la tham do (n nho); ket luan chinh dua vao 3 nhom lon va 'Tat ca'.")