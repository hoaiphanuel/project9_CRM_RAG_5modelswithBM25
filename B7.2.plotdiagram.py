"""
B7.2 - Phân tích & vẽ biểu đồ kết quả thí nghiệm B7.1

    python B7_2_newPlotdiagram.py                 # dùng tau đã log trong B7.1
    python B7_2_newPlotdiagram.py --calibrate     # chọn tau trên nửa dữ liệu, báo cáo trên nửa còn lại (held-out)
    python B7_2_newPlotdiagram.py --show          # hiện cửa sổ biểu đồ (mặc định chỉ lưu PNG)

Đầu ra (thư mục data/figs): fig0_legacy_8plots.png (8 biểu đồ gốc), fig1_overview.png, fig2_heatmaps.png, fig3_abstention.png,
fig4_efficiency_security.png, summary_by_type.csv, summary_by_subtype.csv, pairwise_tests.csv
"""
import argparse
import math
import os
from itertools import combinations

import matplotlib
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--show", action="store_true")
args = ap.parse_args()
if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.getenv("B7_DATA_DIR", "data")
OUT = os.path.join(DATA_DIR, "figs"); os.makedirs(OUT, exist_ok=True)
ORDER = ["rule", "mini", "bge", "e5"]
POS_TYPES = ["Simple", "Ambiguous", "Semantic", "Complex"]
TYPE_ORDER = POS_TYPES + ["Negative"]
sns.set_theme(style="whitegrid", font_scale=0.8)
PAL = dict(zip(ORDER, sns.color_palette("deep", 4)))

df = pd.read_csv(os.path.join(DATA_DIR, "test_results_B7.csv"), keep_default_na=False, na_values=[""])
for c in ["ShouldAbstain", "IsInjection", "HasError", "SecurityViolation", "Answered", "Top1Correct"]:
    df[c] = df[c].astype(str).eq("True")
df["CaseNum"] = df["CaseID"].str[1:].astype(int)
main = df["Type"] != "Security"


# ---------------- hiệu chỉnh ngưỡng tau (post-hoc từ Top1Sim) ----------------
def apply_tau(d, taus):
    d = d.copy()
    sem = d["Model"] != "rule"
    tau = d["Model"].map(taus)
    d.loc[sem, "Answered"] = d.loc[sem, "Top1Sim"].fillna(-1) >= tau[sem]
    ok = d["Top1Correct"]
    d["Hallucination"] = (d["Answered"] & ~ok).astype(float)
    d["EndToEndCorrect"] = np.where(d["ShouldAbstain"], ~d["Answered"], d["Answered"] & ok).astype(float)
    d["AbstainCorrect"] = np.where(d["ShouldAbstain"], (~d["Answered"]).astype(float), np.nan)
    d["FalseAbstain"] = np.where(~d["ShouldAbstain"], (~d["Answered"]).astype(float), np.nan)
    return d


def best_tau(d):
    s = d["Top1Sim"].dropna()
    grid = np.round(np.arange(max(0.0, s.min()), min(1.0, s.max()) + 0.01, 0.01), 2)
    scores = [apply_tau(d, {d["Model"].iloc[0]: t})["EndToEndCorrect"].mean() for t in grid]
    return grid, np.array(scores)


if args.calibrate:
    fit = df[main & (df["CaseNum"] % 2 == 0)]
    taus = {}
    for m in ORDER[1:]:
        g, sc = best_tau(fit[fit["Model"] == m])
        taus[m] = float(g[int(np.argmax(sc))])
    print("tau chọn trên nửa chẵn:", taus)
    df = apply_tau(df, taus)
    df = df[df["CaseNum"] % 2 == 1]                    # chỉ báo cáo trên nửa lẻ (held-out)
    print(f"⚠️  Báo cáo trên {df['CaseID'].nunique()} case held-out (nửa lẻ)")
    main = df["Type"] != "Security"
else:
    taus = None

pos = df[main & ~df["ShouldAbstain"]]
allm = df[main]
sec = df[df["Type"] == "Security"]


# ---------------- bảng tóm tắt ----------------
def summarize(d, by):
    p = d[~d["ShouldAbstain"] & (d["Type"] != "Security")]
    a = d[d["Type"] != "Security"]
    t1 = p.groupby(by + ["Model"])[["Hit@1", "Hit@5", "Recall@5", "MRR", "F1", "ExactMatch", "FalseAbstain"]].mean()
    t2 = a.groupby(by + ["Model"])[["Hallucination", "EndToEndCorrect", "AbstainCorrect"]].mean()
    t3 = d.groupby(by + ["Model"])[["ResponseTime"]].mean()
    return t1.join(t2, how="outer").join(t3, how="outer").round(3).reset_index()


s_type, s_sub = summarize(df, ["Type"]), summarize(df, ["Type", "Subtype"])
s_type.to_csv(os.path.join(OUT, "summary_by_type.csv"), index=False)
s_sub.to_csv(os.path.join(OUT, "summary_by_subtype.csv"), index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print("\n=== TÓM TẮT THEO TYPE ===\n", s_type.to_string(index=False))


# ---------------- kiểm định cặp (McNemar exact) trên EndToEndCorrect ----------------
def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


wide = allm.pivot_table(index="CaseID", columns="Model", values="EndToEndCorrect")
rows = []
for a, b in combinations(ORDER, 2):
    if a in wide and b in wide:
        x, y = wide[a], wide[b]
        n10, n01 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
        rows.append({"A": a, "B": b, "A_only_correct": n10, "B_only_correct": n01,
                     "acc_A": round(x.mean(), 3), "acc_B": round(y.mean(), 3), "p_value": round(mcnemar_p(n10, n01), 4)})
pw = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
pw["p_holm"] = np.minimum(1, np.maximum.accumulate(pw["p_value"].values * (len(pw) - np.arange(len(pw))))).round(4)   # Holm
pw.to_csv(os.path.join(OUT, "pairwise_tests.csv"), index=False)
print("\n=== McNEMAR (EndToEndCorrect, cặp mô hình) ===\n", pw.to_string(index=False))



# ---------------- FIG 0: 8 biểu đồ GỐC (giữ nguyên bố cục 4x2 và tiêu đề của bản cũ) ----------------
# Đối chiếu cột cũ -> cột mới:
#   Correctness -> Hit@1 | Recall@k -> Hit@5 | SemanticScore -> Top1Sim (chỉ 3 mô hình embedding)
#   ExactMatch, F1, ResponseTime, SecurityViolation, InjectionBlocking: giữ tên.
# Lưu ý: 4 biểu đồ chất lượng chỉ tính trên ca CÓ đáp án (Negative không có đáp án nên không nằm ở đây);
#        Security violation chỉ tính trên nhóm Security; Injection Blocking vẽ theo họ payload (Subtype).
def legacy_figure():
    sns.set_theme(style="whitegrid", font_scale=0.7)
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))

    def tbar(ax, d, y, title, ylabel, xlabel="Query Type"):
        sns.barplot(data=d, x="Type", y=y, hue="Model", order=[t for t in POS_TYPES if t in d["Type"].unique()],
                    hue_order=ORDER, palette=PAL, errorbar=None, ax=ax)
        ax.set_title(title, fontsize=9); ax.set_ylabel(ylabel, fontsize=7); ax.set_xlabel(xlabel, fontsize=7)
        ax.legend(title="Model", loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=4, fontsize=6, title_fontsize=7)

    def mbar(ax, d, y, title, ylabel):
        sns.barplot(data=d, x="Model", y=y, hue="Model", order=ORDER, hue_order=ORDER, palette=PAL, errorbar=None, legend=False, ax=ax)
        ax.set_title(title, fontsize=9); ax.set_ylabel(ylabel, fontsize=7); ax.set_xlabel("Model", fontsize=7)

    tbar(axes[0, 0], pos, "Hit@1", "Accuracy by Query Type", "Accuracy")
    tbar(axes[0, 1], pos, "Hit@5", "Recall@k by Query Type", "Recall@k")
    mbar(axes[1, 0], df, "ResponseTime", "Average Response Time by Model", "Time (s)")
    mbar(axes[1, 1], df[df["Model"] != "rule"], "Top1Sim", "Average Semantic Similarity Score", "Score")
    tbar(axes[2, 0], pos, "ExactMatch", "Exact Match by Query Type", "Exact Match")
    tbar(axes[2, 1], pos, "F1", "F1 Score by Query Type", "F1 Score")
    mbar(axes[3, 0], sec, "SecurityViolation", "Security Violation Rate by Model", "Rate")
    inj = sec[sec["IsInjection"]]
    sns.barplot(data=inj, x="Subtype", y="InjectionBlocking", hue="Model", hue_order=ORDER, palette=PAL,
                errorbar=None, ax=axes[3, 1])
    axes[3, 1].set_title("Injection Blocking Rate by Injection Family", fontsize=9)
    axes[3, 1].set_ylabel("Rate", fontsize=7); axes[3, 1].set_xlabel("Injection family", fontsize=7)
    axes[3, 1].tick_params(axis="x", rotation=45)
    axes[3, 1].legend(title="Model", loc="lower center", bbox_to_anchor=(0.5, -0.55), ncol=4, fontsize=6, title_fontsize=7)
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig0_legacy_8plots.png"), dpi=150)
    sns.set_theme(style="whitegrid", font_scale=0.8)


legacy_figure()

# ---------------- FIG 1: tổng quan theo Type ----------------
def bar(ax, d, y, title, types=TYPE_ORDER, ylim=(0, 1.05)):
    d = d[d["Type"].isin(types)]
    sns.barplot(data=d, x="Type", y=y, hue="Model", order=[t for t in types if t in d["Type"].unique()],
                hue_order=ORDER, palette=PAL, errorbar=("ci", 95), capsize=.08, err_kws={"linewidth": 1}, ax=ax)
    ax.set_title(title, fontsize=10); ax.set_xlabel(""); ax.set_ylim(*ylim)
    if ax.get_legend():
        ax.get_legend().remove()


fig, ax = plt.subplots(2, 3, figsize=(15, 8))
bar(ax[0, 0], pos, "Hit@1", "Hit@1 (top-1 đúng) – ca có đáp án", POS_TYPES)
bar(ax[0, 1], pos, "Hit@5", "Hit@5 (đáp án nằm trong top-5)", POS_TYPES)
bar(ax[0, 2], pos, "MRR", "MRR", POS_TYPES)
bar(ax[1, 0], pos, "F1", "Token-F1 (top-1 vs đáp án)", POS_TYPES)
bar(ax[1, 1], allm, "Hallucination", "Hallucination: trả lời sai/không nên trả lời", TYPE_ORDER)
bar(ax[1, 2], allm, "EndToEndCorrect", "End-to-end correct (kể cả ca âm)", TYPE_ORDER)
handles = [plt.Rectangle((0, 0), 1, 1, color=PAL[m]) for m in ORDER]
fig.legend(handles, ORDER, title="Model", loc="lower center", ncol=4, frameon=False)
fig.suptitle("Tổng quan chất lượng truy xuất (thanh lỗi = khoảng tin cậy 95% bootstrap)", fontsize=12)
fig.tight_layout(rect=[0, 0.04, 1, 0.96]); fig.savefig(os.path.join(OUT, "fig1_overview.png"), dpi=150)

# ---------------- FIG 2: heatmap theo Subtype ----------------
fig, ax = plt.subplots(1, 2, figsize=(14, 9), gridspec_kw={"width_ratios": [1, 1]})
h1 = pos.pivot_table(index=["Type", "Subtype"], columns="Model", values="Hit@1")[ORDER]
sns.heatmap(h1, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, cbar=False, ax=ax[0])
ax[0].set_title("Hit@1 theo loại test (ca có đáp án)"); ax[0].set_ylabel("")
h2 = allm.pivot_table(index=["Type", "Subtype"], columns="Model", values="EndToEndCorrect")[ORDER]
sns.heatmap(h2, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax[1])
ax[1].set_title("End-to-end correct (gồm ca âm: đúng = từ chối)"); ax[1].set_ylabel("")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_heatmaps.png"), dpi=150)

# ---------------- FIG 3: khả năng từ chối (tách ca có / không có đáp án) ----------------
sem = allm[(allm["Model"] != "rule") & allm["Top1Sim"].notna()].copy()
sem["Nhóm"] = np.where(sem["ShouldAbstain"], "Không có đáp án", "Có đáp án")


def auc(d):
    r = d["Top1Sim"].rank(); p = d["ShouldAbstain"] == False
    n1, n0 = p.sum(), (~p).sum()
    return (r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


aucs = {m: auc(sem[sem["Model"] == m]) for m in ORDER[1:]}
print("\n=== AUROC: Top1Sim phân biệt 'có đáp án' vs 'không có đáp án' (không phụ thuộc tau) ===")
print({k: round(v, 3) for k, v in aucs.items()})

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
sns.boxplot(data=sem, x="Model", y="Top1Sim", hue="Nhóm", order=ORDER[1:], ax=ax[0], fliersize=2)
for i, m in enumerate(ORDER[1:]):
    t = taus[m] if taus else df.loc[df["Model"] == m, "Tau"].dropna().iloc[0]
    ax[0].hlines(t, i - .4, i + .4, colors="k", linestyles="--", linewidth=1)
ax[0].set_title("Độ tương đồng top-1 (đường đứt = tau)\nAUROC: " + ", ".join(f"{k}={v:.2f}" for k, v in aucs.items()), fontsize=9)
for m in ORDER[1:]:
    g, sc = best_tau(df[main & (df["Model"] == m)])
    ax[1].plot(g, sc, label=m, color=PAL[m])
ax[1].set_title("End-to-end correct theo ngưỡng tau"); ax[1].set_xlabel("tau"); ax[1].legend()
negs = allm[allm["ShouldAbstain"]]
sns.barplot(data=negs, x="Subtype", y="Hallucination", hue="Model", hue_order=ORDER, palette=PAL, errorbar=("ci", 95),
            capsize=.08, err_kws={"linewidth": 1}, ax=ax[2])
ax[2].set_title("False-positive (trả lời khi đáng lẽ từ chối)"); ax[2].tick_params(axis="x", rotation=60); ax[2].set_xlabel("")
ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_abstention.png"), dpi=150)

# ---------------- FIG 4: hiệu năng, độ khó khách hàng, bảo mật ----------------
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
t = df.groupby("Model")[["EncodeTime", "DbTime"]].mean().loc[ORDER] * 1000
t.plot(kind="bar", stacked=True, ax=ax[0, 0], color=["#9bb7d4", "#2f5d8a"]); ax[0, 0].set_ylabel("ms / query")
ax[0, 0].set_title("Thời gian trung bình (encode + DB)"); ax[0, 0].tick_params(axis="x", rotation=0)
sns.barplot(data=pos[pos["CustomerTier"] != ""], x="CustomerTier", y="Hit@1", hue="Model", hue_order=ORDER, palette=PAL,
            order=["10-30", "31-100", ">100"], errorbar=("ci", 95), capsize=.08, err_kws={"linewidth": 1}, ax=ax[0, 1])
ax[0, 1].set_title("Hit@1 theo quy mô khách (số sản phẩm đã mua)"); ax[0, 1].set_ylim(0, 1.05)
sv = sec.pivot_table(index="Subtype", columns="Model", values="SecurityViolation")[ORDER]
sns.heatmap(sv, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1, ax=ax[1, 0], cbar=False)
ax[1, 0].set_title("Security violation (rò rỉ khách khác / bảng bị đổi) – mong muốn = 0"); ax[1, 0].set_ylabel("")
er = sec.pivot_table(index="Subtype", columns="Model", values="HasError")[ORDER].astype(float)
ans = sec[sec["ShouldAbstain"]].pivot_table(index="Subtype", columns="Model", values="Answered")[ORDER].astype(float)
sns.heatmap(ans, annot=True, fmt=".2f", cmap="Oranges", vmin=0, vmax=1, ax=ax[1, 1], cbar=False)
ax[1, 1].set_title("Tỉ lệ vẫn trả lời với payload thuần (mong muốn = 0)"); ax[1, 1].set_ylabel("")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_efficiency_security.png"), dpi=150)

print("\n=== SECURITY ===")
print(sec.groupby("Model")[["SecurityViolation", "HasError", "InjectionBlocking"]].mean().loc[ORDER].round(3))
print("\n=== Lỗi chưa được xử lý (HasError) ===")
print(sec[sec["HasError"]][["Subtype", "Variant", "Model", "Error"]].drop_duplicates().to_string(index=False))
print(f"\n✅ Đã lưu hình & bảng vào {OUT}")
if args.show:
    plt.show()