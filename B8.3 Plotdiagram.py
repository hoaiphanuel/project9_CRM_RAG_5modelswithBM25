"""
B8.3 - Ve bieu do ket qua tu file log cua B8.1 (data/test_results_B7.csv).

Hinh 1 (luoi 4x2, giong phong cach file cu): 8 bieu do tong quan
Hinh 2: heatmap ty le dat theo tung loai con (SubType) x mo hinh

Luu y: can file log do B8.1 (v2) tao ra - co cac cot EvalMode, Success, Hit@K, MRR, Accepted...
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =====================================================
# CAU HINH
# =====================================================
LOG_CSV = "data/test_results_B8.csv"
OUT_DIR = "data/plots"
SAVE = True          # luu PNG (300 dpi) vao OUT_DIR
SHOW = True          # hien cua so bieu do
SHOW_CI = False      # True: ve khoang tin cay 95% (bootstrap) tren cot; khi do bo nhan so tren cot
K_MAIN = 5           # K dung cho "Success" trong B8.1 (chi dung de ghi tieu de)

LABEL = {"rule": "Rule-based", "mini": "MiniLM", "bge": "BGE", "e5": "E5"}
MODEL_ORDER = ["Rule-based", "MiniLM", "BGE", "E5"]
TYPE_ORDER = ["Simple", "Complex", "Ambiguous"]

# =====================================================
# 1. NAP DU LIEU
# =====================================================
df = pd.read_csv(LOG_CSV, dtype={"CaseID": str})
if "EvalMode" not in df.columns:
    raise SystemExit("File log khong co cot EvalMode - hay dung file log do B8.1 (v2) tao ra.")

df["Model"] = df.Model.map(LABEL).fillna(df.Model)
order = [m for m in MODEL_ORDER if m in df.Model.unique()]
ans = df[df.EvalMode != "security"].copy()          # moi ca test
rt = ans[ans.EvalMode == "retrieval"].copy()         # ca truy xuat
ab = ans[ans.EvalMode == "abstain"].copy()           # ca phai tu choi
sec = df[df.EvalMode == "security"].copy()           # kiem tra SQL injection
types = [t for t in TYPE_ORDER if t in ans.Type.unique()]
n_cases = ans.CaseID.nunique()

# =====================================================
# 2. KIEU DANG
# =====================================================
sns.set(style="whitegrid", font_scale=0.7)
PALETTE = dict(zip(MODEL_ORDER, sns.color_palette("deep", len(MODEL_ORDER))))
ERR = ("ci", 95) if SHOW_CI else None


def no_data(ax, msg):
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9)


def drop_legend(ax):
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


def add_labels(ax):
    if not SHOW_CI:
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=5, padding=2)


def bar_by_type(ax, data, y, title, ylabel):
    if data.empty or y not in data.columns or data[y].notna().sum() == 0:
        return no_data(ax, f"No data for {y}")
    sns.barplot(data=data, x="Type", y=y, hue="Model", order=[t for t in types if t in data.Type.unique()],
                hue_order=order, palette=PALETTE, errorbar=ERR, n_boot=1000, ax=ax)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xlabel("Query Type", fontsize=7)
    ax.set_ylim(0, 1.1)
    add_labels(ax)


# =====================================================
# 3. HINH 1 - 8 BIEU DO TONG QUAN
# =====================================================
fig, axes = plt.subplots(4, 2, figsize=(16, 14))

# [0,0] Ty le dat (Success) theo nhom truy van
bar_by_type(axes[0, 0], ans, "Success",
            f"Success Rate by Query Type (hit in top-{K_MAIN}; correct refusal for abstain cases)", "Success rate")
handles, labels = axes[0, 0].get_legend_handles_labels()      # lay chu giai chung cho ca hinh

# [0,1] Hit@K theo K (chi ca truy xuat)
hit_cols = sorted([c for c in rt.columns if c.startswith("Hit@")], key=lambda c: int(c.split("@")[1]))
if hit_cols and len(rt):
    long = rt.melt(id_vars=["Model"], value_vars=hit_cols, var_name="K", value_name="Hit")
    long["K"] = long.K.str.replace("Hit@", "").astype(int)
    sns.lineplot(data=long, x="K", y="Hit", hue="Model", hue_order=order, palette=PALETTE,
                 marker="o", errorbar=None, ax=axes[0, 1])
    axes[0, 1].set_xticks(sorted(long.K.unique()))
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("Hit@K (retrieval cases): at least one correct invoice in top-K", fontsize=9)
    axes[0, 1].set_ylabel("Hit rate", fontsize=7)
    axes[0, 1].set_xlabel("K", fontsize=7)
else:
    no_data(axes[0, 1], "No Hit@K columns in data")

# [1,0] Thoi gian phan hoi trung binh (tach nhung cau truy van va truy van CSDL)
if {"EncodeTime", "DBTime"} <= set(ans.columns):
    t = ans.groupby("Model")[["EncodeTime", "DBTime"]].mean().reindex(order).fillna(0)
    t.plot(kind="bar", stacked=True, ax=axes[1, 0], color=["#8da0cb", "#fc8d62"], rot=0)
    axes[1, 0].set_title("Average Response Time by Model", fontsize=9)
    axes[1, 0].set_ylabel("Time (s)", fontsize=7)
    axes[1, 0].set_xlabel("Model", fontsize=7)
    axes[1, 0].legend(["Query encoding", "Database query"], fontsize=6)
else:
    sns.barplot(data=ans, x="Model", y="ResponseTime", order=order, palette=PALETTE, errorbar=None, ax=axes[1, 0])
    axes[1, 0].set_title("Average Response Time by Model", fontsize=9)
    axes[1, 0].set_ylabel("Time (s)", fontsize=7)

# [1,1] Khoang cach top-1: ca tra loi duoc so voi ca phai tu choi (chi mo hinh embedding)
d = (ans[(ans.Model != "Rule-based") & ans.Top1Distance.notna()].copy()
     if "Top1Distance" in ans.columns else pd.DataFrame())
if len(d):
    d["Case"] = np.where(d.EvalMode == "abstain", "Should abstain", "Answerable")
    sns.boxplot(data=d, x="Model", y="Top1Distance", hue="Case", order=[m for m in order if m != "Rule-based"],
                palette={"Answerable": "#66c2a5", "Should abstain": "#e78ac3"}, showfliers=False, ax=axes[1, 1])
    axes[1, 1].set_title("Top-1 Cosine Distance: answerable vs should-abstain", fontsize=9)
    axes[1, 1].set_ylabel("Distance (lower = more similar)", fontsize=7)
    axes[1, 1].set_xlabel("Model", fontsize=7)
    axes[1, 1].legend(title="", fontsize=6)
else:
    no_data(axes[1, 1], "No Top1Distance in data")

# [2,0] MRR theo nhom truy van (chi ca truy xuat)
bar_by_type(axes[2, 0], rt, "MRR", "Mean Reciprocal Rank by Query Type (retrieval cases)", "MRR")

# [2,1] Ty le tu choi dung o cac ca phai tu choi
if len(ab) and "AbstainOK" in ab.columns:
    sns.barplot(data=ab, x="SubType", y="AbstainOK", hue="Model", hue_order=order, palette=PALETTE,
                errorbar=ERR, n_boot=1000, ax=axes[2, 1])
    axes[2, 1].set_title("Correct Refusal Rate (queries that should be rejected)", fontsize=9)
    axes[2, 1].set_ylabel("Refusal rate", fontsize=7)
    axes[2, 1].set_xlabel("Query sub-type", fontsize=7)
    axes[2, 1].set_ylim(0, 1.1)
    add_labels(axes[2, 1])
else:
    no_data(axes[2, 1], "No abstain cases in data")

# [3,0] An toan truoc SQL injection
if len(sec) and "InjectionSafe" in sec.columns:
    sns.barplot(data=sec, x="Model", y="InjectionSafe", order=order, palette=PALETTE, errorbar=None, ax=axes[3, 0])
    axes[3, 0].set_title("SQL Injection Safety Rate by Model", fontsize=9)
    axes[3, 0].set_ylabel("Rate (1.0 = safe)", fontsize=7)
    axes[3, 0].set_xlabel("Model", fontsize=7)
    axes[3, 0].set_ylim(0, 1.1)
    add_labels(axes[3, 0])
else:
    no_data(axes[3, 0], "No security test rows in data")

# [3,1] Ty le he thong dua ra cau tra loi voi ca tra loi duoc (chi phi cua viec tu choi)
bar_by_type(axes[3, 1], rt, "Accepted", "Answer Rate on Answerable Queries (not wrongly rejected)", "Answer rate")

# Chu giai chung cho ca hinh, bo chu giai rieng cua tung truc
for ax in axes.flat:
    if ax.get_legend() is not None and ax.get_legend().get_title().get_text() == "Model":
        drop_legend(ax)
if labels:
    fig.legend(handles, labels, title="Model", loc="lower center", ncol=len(labels), fontsize=7, title_fontsize=8)
fig.suptitle(f"RAG-CRM retrieval evaluation ({n_cases} test queries)", fontsize=11)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])

# =====================================================
# 4. HINH 2 - HEATMAP THEO LOAI CON
# =====================================================
pv = ans.pivot_table(index=["Type", "SubType"], columns="Model", values="Success", aggfunc="mean")
pv = pv.reindex(columns=order).reset_index()
pv["_t"] = pv.Type.map({t: i for i, t in enumerate(TYPE_ORDER)})
pv = pv.sort_values(["_t", "SubType"]).set_index(["Type", "SubType"]).drop(columns="_t")
pv.index = [f"{t} / {s}" for t, s in pv.index]

fig2, ax2 = plt.subplots(figsize=(7, 0.32 * len(pv) + 1.8))
sns.heatmap(pv, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, linewidths=0.5,
            cbar_kws={"label": "Success rate"}, annot_kws={"size": 6}, ax=ax2)
ax2.set_title("Success Rate by Query Sub-type and Model", fontsize=9)
ax2.set_xlabel("")
ax2.set_ylabel("")
fig2.tight_layout()

# =====================================================
# 5. IN BANG TOM TAT, LUU VA HIEN
# =====================================================
print("=== Success theo nhom truy van ===")
print(ans.pivot_table(index="Type", columns="Model", values="Success", aggfunc="mean")
      .reindex(index=types, columns=order).round(3).to_string())

if SAVE:
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "B7_overview.png"), dpi=300, bbox_inches="tight")
    fig2.savefig(os.path.join(OUT_DIR, "B7_subtype_heatmap.png"), dpi=300, bbox_inches="tight")
    print(f"✅ Da luu bieu do vao {OUT_DIR}/")
if SHOW:
    plt.show()