"""
B9.2 - Phan tich NGUONG TU CHOI (offline, doc data/results_B9.csv do B9_1 tao).

Muc tieu: tra loi "nen chon nguong tu choi the nao?" mot cach co kiem soat, khong nhin tap test khi chon.
Quy trinh (CO DINH TRUOC khi chay):
  * Moi chinh sach chi HOC tren tap dev; ap len tap test mot lan.
  * Muc tieu toi uu = do chinh xac quyet dinh dau-cuoi (end-to-end): ca truy xuat dung neu Hit@5 va duoc tra loi;
    ca phai tu choi dung neu bi tu choi. Tap dev co ty le ca tu choi khac tap test (xem in ra), nen cac ca tu choi
    duoc GAN TRONG SO de ty le = DEPLOY_ABSTAIN_PRIOR (mac dinh 0.20, gan ty le thuc te cua tap test).
  * Chinh sach:
      P0  Youden J, mot nguong chung (cach cu cua B8; chi de doi chung)
      P1  mot nguong chung toi da hoa do chinh xac dau-cuoi tren dev
      P2  nguong theo do dai truy van (<=2 token va >=3 token), moi nhom toi da hoa tren dev
      P3  hoi quy logistic tren dac trung (khoang cach hang 1, bien do hang1-hang2, do dai) -> xac suat "tra loi se dung",
          nguong xac suat chon tren dev
    Chinh sach cua moi phuong phap duoc CHON giua P1-P3 bang kiem tra cheo 5-fold TREN DEV (khong dung test).
  * Bao cao tren test: e2e [95% CI bootstrap], ty le tra loi, ty le tu choi dung, so ca tra loi bia, kiem dinh cap McNemar-Holm.
Dau ra: data/threshold_summary_B9.csv, data/policy_B9.json, data/risk_coverage_B9.png, data/threshold_prior_sensitivity_B9.csv
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

LOG_CSV = "data/results_B9.csv"
OUT_DIR = "data"
DIST_METHODS = ["bm25", "mini", "bge", "e5", "hyb_mini", "hyb_bge", "hyb_e5"]
LABEL = {"rule": "Rule-based", "bm25": "BM25", "mini": "MiniLM", "bge": "BGE", "e5": "E5",
         "hyb_mini": "BM25+MiniLM", "hyb_bge": "BM25+BGE", "hyb_e5": "BM25+E5"}
DEPLOY_ABSTAIN_PRIOR = 0.20      # ty le ca phai tu choi gia dinh khi trien khai (CO DINH TRUOC)
SENSITIVITY_PRIORS = [0.10, 0.20, 0.37, 0.50]
MIN_BUCKET = 8                   # so mau toi thieu cho moi nhom do dai (P2)
SHORT_MAX_TOK = 2
CV_FOLDS = 5
BOOT = 5000
SEED = 42
rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------ du lieu
def load_method(log, method, split, mask=None):
    d = log[(log.Model == method) & (log.Split == split)]
    if mask is not None:
        d = d[mask(d)]
    d = d.sort_values("CaseID").reset_index(drop=True)
    y_ans = (d.EvalMode == "retrieval").values
    hit = np.where(y_ans, d["Hit@5"].fillna(0).values, 0.0) == 1
    d1 = d.Top1Distance.fillna(1.0).values.astype(float)
    d2 = d.Top2Distance.fillna(1.0).values.astype(float)
    ntok = d.NTok.fillna(1).values.astype(float)
    bm = d.Top1BM.fillna(1.0).values.astype(float) if "Top1BM" in d else np.ones(len(d))
    em = d.Top1Emb.fillna(1.0).values.astype(float) if "Top1Emb" in d else np.ones(len(d))
    return dict(df=d, ids=d.CaseID.values, y=y_ans, hit=hit, d1=d1, d2=d2, ntok=ntok, bm=bm, em=em,
                hyb=method.startswith("hyb_"))


def features(D):
    lt = np.log1p(D["ntok"])
    margin = np.clip(D["d2"] - D["d1"], -1, 1)
    if D["hyb"]:
        return np.c_[D["bm"], D["em"], margin, lt, D["em"] * lt]
    return np.c_[D["d1"], margin, lt, D["d1"] * lt]


def sample_weights(y_ans, prior):
    """Cac ca truy xuat trong so 1; cac ca tu choi trong so sao cho (ca tu choi)/(tong) = prior."""
    n_a, n_b = y_ans.sum(), (~y_ans).sum()
    w = np.ones(len(y_ans))
    if n_b > 0 and n_a > 0:
        w[~y_ans] = prior / (1 - prior) * n_a / n_b
    return w


def correct(accept, y_ans, hit):
    return np.where(y_ans, accept & hit, ~accept)


def wacc(accept, D, w):
    return float((correct(accept, D["y"], D["hit"]) * w).sum() / w.sum())


# ------------------------------------------------------------------ cac chinh sach
def candidate_taus(x):
    v = np.unique(x)
    return np.concatenate([[v[0] - 1e-6], (v[:-1] + v[1:]) / 2, [v[-1] + 1e-6]])


def fit_tau_max_acc(x, y_ans, hit, w):
    ts = candidate_taus(x)
    best, bt = -1, ts[0]
    for t in ts:
        acc = x <= t
        a = ((np.where(y_ans, acc & hit, ~acc)) * w).sum() / w.sum()
        if a > best + 1e-12:
            best, bt = a, t
    return float(bt)


def fit_tau_youden(x, y_ans):
    if y_ans.all() or (~y_ans).all():
        return float(np.median(x))
    ts = candidate_taus(x)
    j = [np.mean(x[y_ans] <= t) - np.mean(x[~y_ans] <= t) for t in ts]
    return float(ts[int(np.argmax(j))])


def fit_logreg(X, z, w, l2=1.0, iters=60):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.c_[np.ones(len(X)), (X - mu) / sd]
    beta = np.zeros(A.shape[1])
    R = l2 * np.diag(np.r_[0, np.ones(A.shape[1] - 1)])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ beta, -30, 30)))
        g = A.T @ (w * (z - p)) - R @ beta
        H = A.T @ (A * (w * p * (1 - p) + 1e-9)[:, None]) + R
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-7:
            break
    return dict(mu=mu, sd=sd, beta=beta)


def predict_logreg(m, X):
    A = np.c_[np.ones(len(X)), (X - m["mu"]) / m["sd"]]
    return 1 / (1 + np.exp(-np.clip(A @ m["beta"], -30, 30)))


def fit_policy(name, D, w):
    """Hoc tham so tren D (dev)."""
    if name == "P0":
        return dict(kind="P0", tau=fit_tau_youden(D["d1"], D["y"]))
    if name == "P1":
        return dict(kind="P1", tau=fit_tau_max_acc(D["d1"], D["y"], D["hit"], w))
    if name == "P2":
        g = fit_tau_max_acc(D["d1"], D["y"], D["hit"], w)
        out = dict(kind="P2", tau_global=g, tau_short=g, tau_long=g)
        for key, mask in (("tau_short", D["ntok"] <= SHORT_MAX_TOK), ("tau_long", D["ntok"] > SHORT_MAX_TOK)):
            if mask.sum() >= MIN_BUCKET and D["y"][mask].any() and (~D["y"][mask]).any():
                out[key] = fit_tau_max_acc(D["d1"][mask], D["y"][mask], D["hit"][mask], w[mask])
        return out
    if name == "P3":
        X = features(D)
        m = fit_logreg(X, D["hit"].astype(float), w)
        p = predict_logreg(m, X)
        ts = candidate_taus(p)
        accs = [wacc(p >= t, D, w) for t in ts]
        pstar = float(ts[int(np.argmax(accs))])
        return dict(kind="P3", mu=m["mu"].tolist(), sd=m["sd"].tolist(), beta=m["beta"].tolist(), pstar=pstar)
    raise ValueError(name)


def apply_policy(pol, D):
    """Tra ve (accept, score) ; score cao = tin cay cao."""
    if pol["kind"] in ("P0", "P1"):
        return D["d1"] <= pol["tau"], -D["d1"]
    if pol["kind"] == "P2":
        tau = np.where(D["ntok"] <= SHORT_MAX_TOK, pol["tau_short"], pol["tau_long"])
        return D["d1"] <= tau, -(D["d1"] - tau)
    m = dict(mu=np.array(pol["mu"]), sd=np.array(pol["sd"]), beta=np.array(pol["beta"]))
    p = predict_logreg(m, features(D))
    return p >= pol["pstar"], p


def subset(D, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in D.items() if k != "df"} | {"df": None}


def cv_accuracy(name, D, w, folds=CV_FOLDS):
    n = len(D["y"])
    order = np.arange(n)
    f = np.empty(n, int)
    for cls in (True, False):                      # phan tang theo loai ca
        idx = order[D["y"] == cls]
        rng_ = np.random.default_rng(SEED)
        rng_.shuffle(idx)
        f[idx] = np.arange(len(idx)) % folds
    cor = np.zeros(n, bool)
    for k in range(folds):
        tr, te = np.where(f != k)[0], np.where(f == k)[0]
        pol = fit_policy(name, subset(D, tr), w[tr])
        acc, _ = apply_policy(pol, subset(D, te))
        cor[te] = correct(acc, D["y"][te], D["hit"][te])
    return float((cor * w).sum() / w.sum())


# ------------------------------------------------------------------ do luong tren test
def boot_ci(x):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (BOOT, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def test_metrics(acc, D):
    cor = correct(acc, D["y"], D["hit"])
    ans, ab = D["y"], ~D["y"]
    return dict(e2e=cor.mean(), answer_rate=acc[ans].mean(), refusal=(~acc[ab]).mean(),
                hallucinated=int(acc[ab].sum()), false_refusals=int((~acc[ans]).sum()),
                precision_when_answered=(cor[acc].mean() if acc.any() else np.nan)), cor


def aurc(score, D):
    """Dien tich duoi duong rui ro-do phu: sap xep theo tin cay giam dan; rui ro = ty le sai trong cac ca duoc tra loi."""
    err = ~(D["y"] & D["hit"])                      # tra loi cho ca nay se SAI neu la ca tu choi hoac Hit@5=0
    o = np.argsort(-score, kind="stable")
    e = err[o].astype(float)
    risk = np.cumsum(e) / np.arange(1, len(e) + 1)
    return float(risk.mean()), np.arange(1, len(e) + 1) / len(e), risk


def holm(p):
    o = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def module_report(log, policies):
    """Ap CHINH SACH DA CHON (hoc tren dev) len tung nhom test; bao cao Hit@5, ty le tra loi, tu choi dung, e2e."""
    mods = sorted(m for m in log[log.Split == "test"].Module.unique())
    rows = []
    for mod in mods:
        mk = lambda d, mod=mod: d.Module == mod
        for meth in ["rule"] + list(policies):
            if meth == "rule":
                d = log[(log.Model == "rule") & (log.Split == "test") & (log.Module == mod)].sort_values("CaseID")
                if d.empty:
                    continue
                y = (d.EvalMode == "retrieval").values
                hit = np.where(y, d["Hit@5"].fillna(0).values, 0) == 1
                acc = d.Top5Invoices.notna().values
                D = dict(y=y, hit=hit)
            else:
                D = load_method(log, meth, "test", mk)
                if len(D["y"]) == 0:
                    continue
                info = policies[meth]
                acc, _ = apply_policy(info["params"][info["chosen"]], D)
            cor = correct(acc, D["y"], D["hit"])
            lo, hi = boot_ci(cor)
            n_r, n_a = int(D["y"].sum()), int((~D["y"]).sum())
            rows.append(dict(Module=mod, Method=meth, n=len(cor), n_answerable=n_r, n_out_of_scope=n_a,
                             Hit5=D["hit"][D["y"]].mean() if n_r else np.nan,
                             answer_rate=acc[D["y"]].mean() if n_r else np.nan,
                             correct_refusal=(~acc[~D["y"]]).mean() if n_a else np.nan,
                             e2e=cor.mean(), e2e_lo=lo, e2e_hi=hi))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/module_summary_B9.csv", index=False)
    print("\n=== THEO NHOM TEST (chinh sach da chon tren dev) ===")
    for col, title in (("Hit5", "Hit@5 (ca truy xuat)"), ("answer_rate", "Ty le tra loi (ca truy xuat)"),
                       ("correct_refusal", "Ty le tu choi dung (ca phai tu choi)"), ("e2e", "Quyet dinh dung dau-cuoi")):
        pv = df.pivot_table(index="Module", columns="Method", values=col)
        pv = pv.reindex(columns=[m for m in ["rule"] + DIST_METHODS if m in pv.columns])
        print(f"\n-- {title} --")
        print(pv.round(3).to_string())
    n_tab = df[df.Method == "bm25"][["Module", "n", "n_answerable", "n_out_of_scope"]]
    print("\nSo ca moi nhom:\n" + n_tab.to_string(index=False))


def main():
    log = pd.read_csv(LOG_CSV, dtype={"CaseID": str})
    if "Module" not in log.columns:
        log["Module"] = "core"
    log["Module"] = log.Module.fillna("core").astype(str)
    dev_mask = lambda d: ~d.Module.str.endswith("_Q")            # chon nguong tren dev cua tat ca nhom o che do cum tim
    core_mask = lambda d: d.Module == "core"                     # bao cao chinh tren 500 ca test cua B8
    dev_n = log[(log.Split == "dev") & (log.Model == "bm25") & dev_mask(log)]
    tst_n = log[(log.Split == "test") & (log.Model == "bm25") & core_mask(log)]
    print(f"Dev: {len(dev_n)} ca ({(dev_n.EvalMode == 'abstain').mean():.0%} phai tu choi) | "
          f"Test: {len(tst_n)} ca ({(tst_n.EvalMode == 'abstain').mean():.0%} phai tu choi)")
    print(f"Trong so ca tu choi trong dev duoc dua ve ty le {DEPLOY_ABSTAIN_PRIOR:.0%}\n")
    if len(dev_n) < 60:
        print("⚠️  Tap dev nho (<60 ca): ket qua chon chinh sach kem on dinh")

    rows, policies, curves, test_cor = [], {}, {}, {}
    # --- baseline rule: tra loi khi co dong nao
    R = log[(log.Model == "rule") & (log.Split == "test") & core_mask(log)].sort_values("CaseID")
    y = (R.EvalMode == "retrieval").values
    hit = np.where(y, R["Hit@5"].fillna(0).values, 0) == 1
    acc = R.Top5Invoices.notna().values
    Dr = dict(y=y, hit=hit)
    m, cor = test_metrics(acc, Dr)
    lo, hi = boot_ci(cor)
    rows.append(dict(Method="rule", Policy="co dong nao thi tra loi", dev_cv_acc=np.nan, test_e2e=m["e2e"], ci_lo=lo, ci_hi=hi, **{k: v for k, v in m.items() if k != "e2e"}))
    test_cor[("rule", "baseline")] = pd.Series(cor, index=R.CaseID.values)

    for meth in DIST_METHODS:
        if not ((log.Model == meth) & (log.Split == "dev")).any():
            print(f"(bo qua {meth}: khong co du lieu dev)")
            continue
        Dv, Dt = load_method(log, meth, "dev", dev_mask), load_method(log, meth, "test", core_mask)
        w = sample_weights(Dv["y"], DEPLOY_ABSTAIN_PRIOR)
        cvs = {p: cv_accuracy(p, Dv, w) for p in ("P1", "P2", "P3")}
        chosen = max(cvs, key=lambda p: (round(cvs[p], 6), -["P1", "P2", "P3"].index(p)))   # hoa: uu tien don gian hon
        pols = {p: fit_policy(p, Dv, w) for p in ("P0", "P1", "P2", "P3")}
        policies[meth] = dict(chosen=chosen, cv_dev=cvs, params=pols)
        print(f"{LABEL[meth]:12s} CV dev: " + ", ".join(f"{p}={v:.3f}" for p, v in cvs.items()) + f"  -> chon {chosen}")
        for p in ("P0", "P1", "P2", "P3"):
            acc, score = apply_policy(pols[p], Dt)
            m, cor = test_metrics(acc, Dt)
            lo, hi = boot_ci(cor)
            a_, _, _ = aurc(score, Dt)
            rows.append(dict(Method=meth, Policy=p + (" (da chon)" if p == chosen else ""), dev_cv_acc=cvs.get(p, np.nan),
                             test_e2e=m["e2e"], ci_lo=lo, ci_hi=hi, AURC=a_,
                             **{k: v for k, v in m.items() if k != "e2e"}))
            test_cor[(meth, p)] = pd.Series(cor, index=Dt["ids"])
        a_d, cov_d, risk_d = aurc(-Dt["d1"], Dt)
        a_l, cov_l, risk_l = aurc(apply_policy(pols["P3"], Dt)[1], Dt)
        curves[meth] = (cov_d, risk_d, cov_l, risk_l, a_d, a_l)

    summ = pd.DataFrame(rows)
    summ.to_csv(f"{OUT_DIR}/threshold_summary_B9.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== KET QUA TREN TEST (chinh sach hoc tren dev) ===")
    show = summ.copy()
    show["Method"] = show.Method.map(LABEL)
    print(show.round(3).to_string(index=False))

    # --- so sanh cap giua cac phuong phap voi chinh sach DA CHON (McNemar, Holm)
    sel = {m: test_cor[(m, policies[m]["chosen"])] for m in policies}
    sel["rule"] = test_cor[("rule", "baseline")]
    names = list(sel)
    prs = list(itertools.combinations(names, 2))
    ps, rs = [], []
    for a, b in prs:
        ca, cb = sel[a].align(sel[b], join="inner")
        n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        ps.append(stats.binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0)
        rs.append((LABEL[a], LABEL[b], ca.mean() - cb.mean(), n10, n01))
    ph = holm(np.array(ps))
    pw = pd.DataFrame(rs, columns=["A", "B", "Chenh_e2e", "A_dung_B_sai", "A_sai_B_dung"])
    pw["p"], pw["p_Holm"] = ps, ph
    pw.to_csv(f"{OUT_DIR}/threshold_pairwise_B9.csv", index=False)
    print("\n=== SO SANH CAP (chinh sach da chon, e2e, McNemar + Holm) - cac cap co p_Holm<0.05 ===")
    print(pw[pw.p_Holm < 0.05].round(4).to_string(index=False) or "(khong co cap nao)")

    # --- do nhay theo ty le ca tu choi gia dinh
    sens = []
    for meth in policies:
        Dv, Dt = load_method(log, meth, "dev", dev_mask), load_method(log, meth, "test", core_mask)
        for pr in SENSITIVITY_PRIORS:
            pol = fit_policy("P1", Dv, sample_weights(Dv["y"], pr))
            m, _ = test_metrics(apply_policy(pol, Dt)[0], Dt)
            sens.append(dict(Method=LABEL[meth], prior=pr, tau=pol["tau"], **m))
    pd.DataFrame(sens).to_csv(f"{OUT_DIR}/threshold_prior_sensitivity_B9.csv", index=False)
    print("\n=== P1: do nhay theo ty le ca tu choi gia dinh (e2e tren test) ===")
    print(pd.DataFrame(sens).pivot(index="Method", columns="prior", values="e2e").round(3).to_string())

    module_report(log, policies)

    # --- luu chinh sach de B9_3 dung
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    json.dump(conv(dict(prior=DEPLOY_ABSTAIN_PRIOR, methods=policies)), open(f"{OUT_DIR}/policy_B9.json", "w"), indent=1)

    # --- ve duong rui ro - do phu
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for meth, (cd, rd, cl, rl, ad, al) in curves.items():
        ax[0].plot(cd, rd, label=f"{LABEL[meth]} (AURC {ad:.3f})")
        ax[1].plot(cl, rl, label=f"{LABEL[meth]} (AURC {al:.3f})")
    ax[0].set_title("Chi dung khoang cach hang 1")
    ax[1].set_title("Hoi quy logistic (khoang cach + bien do + do dai)")
    for a in ax:
        a.set_xlabel("Do phu (ty le truy van duoc tra loi)")
        a.legend(fontsize=7)
        a.grid(alpha=.3)
    ax[0].set_ylabel("Rui ro (ty le sai trong cac ca duoc tra loi)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/risk_coverage_B9.png", dpi=300)
    print(f"\n✅ Da luu: threshold_summary_B9.csv, threshold_pairwise_B9.csv, threshold_prior_sensitivity_B9.csv, policy_B9.json, risk_coverage_B9.png")


if __name__ == "__main__":
    main()
