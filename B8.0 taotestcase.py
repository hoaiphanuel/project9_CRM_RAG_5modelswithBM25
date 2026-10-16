"""
B8.0 (v2) - Sinh bo test da dang cho danh gia truy xuat RAG-CRM tren Online Retail II.

Nguyen tac: moi ca deu CHAY DUOC bang chinh cach he thong lam (tim theo mo ta + loc theo khach),
va dap an chuan luon TINH TU DU LIEU GOC (khong lay tu dau ra cua he thong).

Dau ra: data/test_cases_B7.csv
  QueryID, Split (test | dev), Type, SubType, EvalMode (retrieval | abstain), Query, CustomerID,
  SearchText, ExpectedInvoiceNos ("|"), NumExpected, ExpectedAnswerText
  - Split = test : 150 ca (50 Simple + 50 Complex + 50 Ambiguous) de bao cao ket qua
  - Split = dev  : 60 ca rieng biet, chi dung de chon nguong tu choi (khong bao cao)
  - retrieval : dung neu top-K co it nhat 1 hoa don thuoc ExpectedInvoiceNos
  - abstain   : he thong phai tu choi (ExpectedAnswerText = NOT_FOUND)
"""
import os
import random
import re
from collections import Counter

import pandas as pd

SEED = 42
INPUT_CSV = "data/online_retail_clean.csv"
OUTPUT_CSV = "data/test_cases_B8.csv"
random.seed(SEED)

# =====================================================
# 1. NAP VA CHUAN HOA DU LIEU
# =====================================================
raw = pd.read_csv(INPUT_CSV, dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str})
need = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
missing = [c for c in need if c not in raw.columns]
if missing:
    raise SystemExit(f"Thieu cot: {missing}. Cac cot hien co: {list(raw.columns)}")

A = raw.dropna(subset=["CustomerID", "Description"]).copy()
A["CustomerID"] = A["CustomerID"].astype(float).astype(int)
A["Description"] = A["Description"].astype(str).str.strip()
try:
    A["InvoiceDate"] = pd.to_datetime(A["InvoiceDate"], format="%m/%d/%Y %H:%M")
except ValueError:
    A["InvoiceDate"] = pd.to_datetime(A["InvoiceDate"])
A = A[~A.InvoiceNo.str.startswith("C") & (A.Quantity > 0) & (A.UnitPrice > 0)].copy()
A["LineTotal"] = A["Revenue"] if "Revenue" in A.columns else A.Quantity * A.UnitPrice
A["Day"] = A.InvoiceDate.dt.normalize()

# P = chi dong SAN PHAM that (bo phi van chuyen, dieu chinh, hang test, ky tu dac biet)
NP_DESC = {"postage", "manual", "carriage", "discount", "bank charges", "this is a test product."}
sc = A.StockCode.str.upper()
dl = A.Description.str.lower()
is_np = (dl.isin(NP_DESC) | dl.str.startswith("adjustment")
         | sc.isin(["POST", "M", "C2", "D", "DOT", "BANK CHARGES", "CRUK", "AMAZONFEE", "S", "B"])
         | sc.str.startswith(("ADJUST", "TEST")))
bad_text = (A.Description.str.contains(r"[%_\\]")
            | (A.Description.str.contains("'") & A.Description.str.contains('"')))
P = A[~is_np & ~bad_text].copy()
print(f"✅ {len(A):,} dong | {len(P):,} dong san pham | {A.CustomerID.nunique():,} khach | "
      f"{A.InvoiceNo.nunique():,} hoa don | {A.InvoiceDate.min():%Y-%m-%d} -> {A.InvoiceDate.max():%Y-%m-%d}")

P_by_c = {k: g for k, g in P.groupby("CustomerID")}
P_by_inv = {k: g for k, g in P.groupby("InvoiceNo")}
INV_TOTAL = A.groupby("InvoiceNo").LineTotal.sum()
ALL_DESC = P.Description.unique().tolist()
ALL_DESC_SET = set(ALL_DESC)
WORD_DF = Counter(w for d in ALL_DESC for w in set(re.findall(r"[a-z]{4,}", d.lower())))


# =====================================================
# 2. TIEN ICH
# =====================================================
def qt(s):
    return f"'{s}'" if '"' in s else f'"{s}"'


def dq(d):
    return random.choice([f"{d:%Y-%m-%d}", f"{d:%d %B %Y}", f"{d:%B %d, %Y}"])


def inv_str(df_):
    return "|".join(sorted(df_.InvoiceNo.unique()))


def pair_lines(c, d):
    g = P_by_c[c]
    return g[g.Description == d]


def rec(typ, sub, mode, q, ans, inv="", cid="", text=""):
    return {"Type": typ, "SubType": sub, "EvalMode": mode, "Query": q, "CustomerID": cid,
            "SearchText": text, "ExpectedInvoiceNos": inv,
            "NumExpected": len([x for x in inv.split("|") if x]), "ExpectedAnswerText": ans}


def perturb(text):
    for _ in range(5):
        words = text.split()
        mode = random.choice(["drop", "typo", "half"])
        if mode == "drop" and len(words) > 2:
            words.pop(random.randrange(len(words)))
        elif mode == "typo":
            i = random.randrange(len(words))
            w = words[i]
            if len(w) > 3:
                j = random.randrange(len(w) - 1)
                words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        else:
            words = words[:max(1, len(words) // 2)] + ["something", "similar"]
        out = " ".join(words)
        if out != text:
            return out
    return text + " thing"


SYN = {"mug": "cup", "cushion": "pillow", "lights": "lamps", "light": "lamp", "decoration": "ornament",
       "christmas": "xmas", "vintage": "retro", "large": "big", "small": "little", "holder": "stand",
       "bottle": "flask", "bowl": "dish", "box": "case", "cover": "case", "design": "pattern",
       "notebook": "journal", "bag": "pouch"}


def has_syn(d):
    return any(w in SYN for w in re.findall(r"[a-z]+", d.lower()))


def synonymize(d):
    words, changed = d.split(), 0
    for i, w in enumerate(words):
        core = re.sub(r"[^a-z]", "", w.lower())
        if core in SYN and changed < 2:
            words[i] = SYN[core]
            changed += 1
    return " ".join(words)


# =====================================================
# 3. SIMPLE - tim san pham khach da mua
# =====================================================
def s_exact(r):
    c, d = r.CustomerID, r.Description
    q = random.choice([
        "Did customer {c} purchase the product {d}?", "Has customer {c} ever ordered {d}?",
        "Is {d} in the purchase history of customer {c}?", "Check whether customer {c} bought {d}.",
        "Customer {c} - did they buy {d} before?", "Can you confirm that customer {c} purchased {d}?",
        "which invoice has {d} for customer {c}", "{d} customer {c}"]).format(c=c, d=qt(d))
    return rec("Simple", "bought_exact", "retrieval", q, f"Customer {c} bought {d}",
               inv_str(pair_lines(c, d)), cid=c, text=d)


def s_short(r):
    c = r.CustomerID
    t = " ".join(r.Description.split()[:2])
    if len(t) < 5:
        return None
    g = P_by_c[c]
    sub = g[g.Description.str.lower().str.startswith(t.lower())]
    q = random.choice(["Customer {c} bought something starting with {t}. Which invoice?",
                       "Find {t} items in the orders of customer {c}.",
                       "Look up {t} for customer {c}."]).format(c=c, t=qt(t))
    return rec("Simple", "bought_short_name", "retrieval", q, f"Customer {c} bought products starting with {t}",
               inv_str(sub), cid=c, text=t)


def s_case(r):
    c, d = r.CustomerID, r.Description
    text = random.choice([d.upper(), d.title(), d.replace(" ", "  ")])
    q = random.choice(["Did customer {c} buy {t}?", "Customer {c}: {t} - any invoice?"]).format(c=c, t=qt(text))
    return rec("Simple", "bought_case_variant", "retrieval", q, f"Customer {c} bought {d}",
               inv_str(pair_lines(c, d)), cid=c, text=text)


def s_month(r):
    c, d = r.CustomerID, r.Description
    sub = pair_lines(c, d)
    ref = pd.Timestamp(random.choice(list(sub.Day.unique())))
    ok = sub[sub.Day.dt.to_period("M") == ref.to_period("M")]
    q = random.choice(["Did customer {c} buy {d} in {m}?",
                       "Find the invoice of customer {c} for {d} during {m}."]).format(c=c, d=qt(d), m=f"{ref:%B %Y}")
    return rec("Simple", "bought_in_month", "retrieval", q, f"Customer {c} bought {d} in {ref:%B %Y}",
               inv_str(ok), cid=c, text=d)


def s_not_bought(r):
    c = r.CustomerID
    bought = set(P_by_c[c].Description)
    d = random.choice(ALL_DESC)
    while d in bought:
        d = random.choice(ALL_DESC)
    q = random.choice(["Did customer {c} purchase the product {d}?", "Has customer {c} ever ordered {d}?"]).format(
        c=c, d=qt(d))
    return rec("Simple", "not_bought", "abstain", q, "NOT_FOUND", cid=c, text=d)


# =====================================================
# 4. COMPLEX - san pham + rang buoc (so luong, gia tri, ngay, lap lai...)
# Dap an chuan = cac hoa don THOA rang buoc, tinh tu du lieu.
# =====================================================
def _tot(sub):
    return sub.InvoiceNo.map(INV_TOTAL)


def x_qty_gt(r):
    c, d = r.CustomerID, r.Description
    sub = pair_lines(c, d)
    feas = [n for n in (5, 10, 20, 50, 100) if (sub.Quantity > n).any()]
    if not feas:
        return None
    n = random.choice(feas)
    q = random.choice(["Find invoices of customer {c} containing {d} with quantity greater than {n}.",
                       "Did customer {c} ever order more than {n} units of {d} at once?"]).format(
        c=c, d=qt(d), n=n)
    return rec("Complex", "quantity_gt", "retrieval", q, f"Customer {c}: {d} with quantity > {n}",
               inv_str(sub[sub.Quantity > n]), cid=c, text=d)


def x_value_gt(r):
    c, d = r.CustomerID, r.Description
    sub = pair_lines(c, d)
    t = _tot(sub)
    feas = [v for v in (100, 200, 500, 1000) if (t > v).any()]
    if not feas:
        return None
    v = random.choice(feas)
    q = random.choice(["Is there an invoice of customer {c} with {d} whose order total is above {v}?",
                       "Find the orders of customer {c} that include {d} and are worth more than {v}."]).format(
        c=c, d=qt(d), v=v)
    return rec("Complex", "order_value_gt", "retrieval", q, f"Customer {c}: {d} in order with total > {v}",
               inv_str(sub[t > v]), cid=c, text=d)


def x_qty_or_value(r):
    c, d = r.CustomerID, r.Description
    sub = pair_lines(c, d)
    ok = sub[(sub.Quantity > 5) | (_tot(sub) > 100)]
    if ok.empty:
        return None
    q = random.choice([
        "Is there any invoice of customer {c} containing {d} with quantity greater than 5 or total order value above 100?",
        "Find the invoices of customer {c} that include {d} where quantity > 5 or the order total exceeds 100."]).format(
        c=c, d=qt(d))
    return rec("Complex", "quantity_or_value", "retrieval", q,
               f"Customer {c}: {d} (quantity > 5 or value > 100)", inv_str(ok), cid=c, text=d)


def x_qty_and_value(r):
    c, d = r.CustomerID, r.Description
    sub = pair_lines(c, d)
    ok = sub[(sub.Quantity >= 5) & (_tot(sub) > 100)]
    if ok.empty:
        return None
    q = random.choice(["Find invoices of customer {c} with {d} where quantity is at least 5 and the order total is above 100.",
                       "Did customer {c} place a big order (5+ units, total over 100) including {d}?"]).format(
        c=c, d=qt(d))
    return rec("Complex", "quantity_and_value", "retrieval", q,
               f"Customer {c}: {d} (quantity >= 5 and value > 100)", inv_str(ok), cid=c, text=d)


def x_date_window(r):
    c, d = r.CustomerID, r.Description
    sub = pair_lines(c, d)
    end = pd.Timestamp(random.choice(list(sub.Day.unique())))
    start = end - pd.Timedelta(days=random.choice([90, 180]))
    ok = sub[(sub.Day >= start) & (sub.Day <= end)]
    q = random.choice(["Did customer {c} buy {d} between {s} and {e}?",
                       "Find invoices of customer {c} for {d} from {s} to {e}."]).format(
        c=c, d=qt(d), s=dq(start), e=dq(end))
    return rec("Complex", "date_window", "retrieval", q, f"Customer {c}: {d} between {start:%Y-%m-%d} and {end:%Y-%m-%d}",
               inv_str(ok), cid=c, text=d)


def x_latest(r):
    c, d = r.CustomerID, r.Description
    sub = pair_lines(c, d).sort_values("InvoiceDate")
    if sub.InvoiceNo.nunique() < 2:
        return None
    last = sub[sub.InvoiceDate == sub.InvoiceDate.max()]
    if last.InvoiceNo.nunique() != 1:
        return None
    q = random.choice(["What was the most recent invoice where customer {c} bought {d}?",
                       "When did customer {c} last order {d}? Give the invoice."]).format(c=c, d=qt(d))
    return rec("Complex", "latest_purchase", "retrieval", q, f"Latest: invoice {last.InvoiceNo.iloc[0]}",
               last.InvoiceNo.iloc[0], cid=c, text=d)


def x_repeat(r):
    c, d = r.CustomerID, r.Description
    sub = pair_lines(c, d)
    if sub.InvoiceNo.nunique() < 2:
        return None
    q = random.choice(["Did customer {c} buy {d} more than once? List the invoices.",
                       "Show every order where customer {c} repurchased {d}."]).format(c=c, d=qt(d))
    return rec("Complex", "repeat_purchase", "retrieval", q, f"Customer {c} bought {d} repeatedly",
               inv_str(sub), cid=c, text=d)


def x_cooccur(r):
    c, a = r.CustomerID, r.Description
    others = [x for x in P_by_inv[r.InvoiceNo].Description.unique() if x != a]
    if not others:
        return None
    b = random.choice(others)
    both = set(pair_lines(c, a).InvoiceNo) & set(pair_lines(c, b).InvoiceNo)
    if not both:
        return None
    q = random.choice(["Did customer {c} buy both {a} and {b} in the same invoice?",
                       "Find the invoice of customer {c} that contains {a} together with {b}."]).format(
        c=c, a=qt(a), b=qt(b))
    return rec("Complex", "same_invoice_pair", "retrieval", q, f"Invoice {'|'.join(sorted(both))} has both",
               "|".join(sorted(both)), cid=c, text=f"{a} and {b}")


# =====================================================
# 5. AMBIGUOUS - nhieu, mo ho, khong ton tai
# =====================================================
def a_typo(r):
    c, d = r.CustomerID, r.Description
    noisy = perturb(d)
    q = random.choice(["Find an invoice of customer {c} for a product like {n}.",
                       "Customer {c} bought something called {n}, which invoice?",
                       "I don't remember the exact name, maybe {n}. Customer {c}."]).format(c=c, n=qt(noisy))
    return rec("Ambiguous", "typo_or_partial", "retrieval", q, f"Customer {c} has an invoice with {d}",
               inv_str(pair_lines(c, d)), cid=c, text=noisy)


def a_keyword(r):
    c = r.CustomerID
    words = re.findall(r"[a-z]{4,}", r.Description.lower())
    if not words:
        return None
    word = min(words, key=lambda w: WORD_DF[w])
    g = P_by_c[c]
    sub = g[g.Description.str.lower().str.contains(word, regex=False)]
    if sub.InvoiceNo.nunique() > 15:
        return None
    q = random.choice(["Customer {c} bought something with {w} in the name. Which invoice(s)?",
                       "Which of customer {c}'s invoices include an item related to {w}?"]).format(c=c, w=qt(word))
    return rec("Ambiguous", "keyword_only", "retrieval", q, f"Invoices containing {word}",
               inv_str(sub), cid=c, text=word)


def a_synonym(r):
    c, d = r.CustomerID, r.Description
    s = synonymize(d)
    if s == d:
        return None
    q = random.choice(["Find an invoice of customer {c} for {s}.",
                       "Did customer {c} order something like {s}?"]).format(c=c, s=qt(s))
    return rec("Ambiguous", "synonym", "retrieval", q, f"Customer {c} has an invoice with {d}",
               inv_str(pair_lines(c, d)), cid=c, text=s)


PREFIX = ["something like", "i think it was", "the one called", "maybe", "not sure but", "abc", "test", "random"]
SUFFIX = ["or similar product", "or something similar", "or some accessory", "i guess", ""]


def a_noise(r):
    c, d = r.CustomerID, r.Description
    text = " ".join(x for x in [random.choice(PREFIX), d, random.choice(SUFFIX)] if x)
    q = random.choice(["Find an invoice of customer {c} with description {t}.",
                       "Customer {c} - look for {t}."]).format(c=c, t=qt(text))
    return rec("Ambiguous", "noise_phrase", "retrieval", q, f"Customer {c} has an invoice with {d}",
               inv_str(pair_lines(c, d)), cid=c, text=text)


def a_bilingual(r):
    c, d = r.CustomerID, r.Description
    noisy = perturb(d)
    q, text = random.choice([
        (f"Khách {c} có mua {qt(d)} hoặc món tương tự không?", d),
        (f"Tìm hóa đơn của khách {c} có sản phẩm giống {qt(noisy)}.", noisy),
        (f"Khách hàng {c} đã mua món nào tên {qt(noisy)} vậy?", noisy)])
    return rec("Ambiguous", "bilingual", "retrieval", q, f"Customer {c} bought {d}",
               inv_str(pair_lines(c, d)), cid=c, text=text)


def a_no_customer(r):
    d = r.Description
    noisy = perturb(d)
    sub = P[P.Description == d]
    q = random.choice(["Which invoice contains a product like {n}?", "Who bought something like {n}?"]).format(
        n=qt(noisy))
    return rec("Ambiguous", "missing_customer", "retrieval", q,
               "Customers: " + ", ".join(map(str, sorted(sub.CustomerID.unique()))), inv_str(sub), cid="", text=noisy)


ADJ = ["unicorn", "quantum", "golden", "holographic", "diamond", "solar", "velvet", "antique", "neon", "crystal"]
NOUN = ["teapot", "toaster oven", "garden gnome", "lunch box", "lava lamp", "cuckoo clock", "candle set", "mug set"]


def a_unanswerable(r):
    for _ in range(20):
        f = f"{random.choice(ADJ)} {random.choice(NOUN)}"
        if f not in ALL_DESC_SET:
            break
    q = random.choice(["Did customer {c} purchase {f}?", "Find the invoice where customer {c} bought {f}."]).format(
        c=r.CustomerID, f=qt(f))
    return rec("Ambiguous", "not_found", "abstain", q, "NOT_FOUND", cid=r.CustomerID, text=f)


# =====================================================
# 6. CAC "HO" MAU
# =====================================================
pair = P.drop_duplicates(["CustomerID", "Description"]).copy()
pair["_key"] = "p:" + pair.CustomerID.astype(str) + "|" + pair.Description
n_inv_pair = P.groupby(["CustomerID", "Description"]).InvoiceNo.nunique()
pair["_ninv"] = [n_inv_pair[(c, d)] for c, d in zip(pair.CustomerID, pair.Description)]
nw = pair.Description.str.split().str.len()
inv_per_desc = P.groupby("Description").InvoiceNo.nunique()

pool_pair = pair
pool_multi = pair[nw >= 3]
pool_rare = pool_multi[pool_multi.Description.map(inv_per_desc).between(1, 5)]
pool_syn = pool_multi[pool_multi.Description.map(has_syn)]
pool_rep = pair[pair._ninv >= 2]

# =====================================================
# 7. KE HOACH
# =====================================================
test_plan = [
    # Simple = 50
    (s_exact, pool_pair, 12), (s_short, pool_multi, 10), (s_case, pool_pair, 10),
    (s_month, pool_pair, 8), (s_not_bought, pool_pair, 10),
    # Complex = 50
    (x_qty_gt, pool_pair, 8), (x_value_gt, pool_pair, 7), (x_qty_or_value, pool_pair, 8),
    (x_qty_and_value, pool_pair, 5), (x_date_window, pool_pair, 7), (x_latest, pool_rep, 5),
    (x_repeat, pool_rep, 5), (x_cooccur, pool_pair, 5),
    # Ambiguous = 50
    (a_typo, pool_multi, 10), (a_keyword, pool_multi, 8), (a_synonym, pool_syn, 8),
    (a_noise, pool_multi, 8), (a_bilingual, pool_multi, 6), (a_no_customer, pool_rare, 5),
    (a_unanswerable, pool_pair, 5),
]
# Tap dev (chi de chon nguong tu choi): cung loai ca nhung cap (khach, san pham) khac hoan toan
dev_plan = [
    (s_exact, pool_pair, 8), (s_short, pool_multi, 3), (s_case, pool_pair, 3),
    (x_qty_or_value, pool_pair, 5), (x_date_window, pool_pair, 3),
    (a_typo, pool_multi, 5), (a_noise, pool_multi, 4), (a_synonym, pool_syn, 3), (a_keyword, pool_multi, 4),
    (s_not_bought, pool_pair, 12), (a_unanswerable, pool_pair, 10),
]

used = set()


def build(plan, seed0, split):
    out = []
    for i, (gen, pool, n) in enumerate(plan):
        got = []
        cand = pool[~pool._key.isin(used)].sample(frac=1, random_state=seed0 + i)
        for _, row in cand.iterrows():
            res = gen(row)
            if res is None:
                continue
            used.add(row._key)
            res["Split"] = split
            got.append(res)
            if len(got) == n:
                break
        if len(got) < n:
            print(f"⚠️  {gen.__name__}: chi tao duoc {len(got)}/{n} ca")
        out += got
    return out


test_cases = build(test_plan, SEED, "test")
dev_cases = build(dev_plan, SEED + 1000, "dev")
out = pd.DataFrame(test_cases + dev_cases).drop_duplicates("Query").reset_index(drop=True)

prefix = {"Simple": "S", "Complex": "C", "Ambiguous": "A"}
ids, cnt = [], Counter()
for _, r in out.iterrows():
    key = "D" if r.Split == "dev" else prefix[r.Type]
    cnt[key] += 1
    ids.append(f"{key}{cnt[key]:03d}")
out.insert(0, "QueryID", ids)
out = out[["QueryID", "Split", "Type", "SubType", "EvalMode", "Query", "CustomerID", "SearchText",
           "ExpectedInvoiceNos", "NumExpected", "ExpectedAnswerText"]]

# =====================================================
# 8. KIEM TRA CHAT LUONG
# =====================================================
r_ = out[out.EvalMode == "retrieval"]
a_ = out[out.EvalMode == "abstain"]
assert (r_.NumExpected >= 1).all(), r_[r_.NumExpected < 1].QueryID.tolist()
assert (a_.NumExpected == 0).all()
assert (out.SearchText != "").all()

os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("\n" + out.groupby(["Split", "Type", "SubType", "EvalMode"]).size().to_string())
t = out[out.Split == "test"]
print(f"\nTest: {len(t)} ca | Dev: {(out.Split == 'dev').sum()} ca | abstain trong test: {(t.EvalMode == 'abstain').sum()}")
print(f"Ca retrieval (test) co nhieu hoa don dung: {(t[t.EvalMode == 'retrieval'].NumExpected > 1).mean():.0%}")
print("\nVi du moi loai con:")
for _, r in t.drop_duplicates("SubType").iterrows():
    print(f"  [{r.SubType}] {r.Query}  ->  {r.NumExpected} hoa don dung")
print(f"\n🎉 Da tao {OUTPUT_CSV}")