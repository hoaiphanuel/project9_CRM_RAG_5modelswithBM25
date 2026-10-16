"""
B7.0 - Sinh bộ test case đa dạng cho so sánh Rule-based vs 3 embedding models.

Nguyên tắc thiết kế
-------------------
1. Ground truth là TẬP các sản phẩm (Description) hợp lệ, tính trực tiếp bằng pandas
   từ dữ liệu gốc  ->  không còn tình trạng "đáp án chỉ có 1 InvoiceNo" trong khi khách
   mua cùng sản phẩm ở nhiều hoá đơn.
2. Mỗi test case gồm SearchText (chuỗi thực sự đưa vào engine) + các bộ lọc có cấu trúc
   (QtyGt, InvoiceTotalGt, DateFrom, DateTo) -> phần "Complex" thực sự được kiểm thử.
3. Có test case ÂM (khách chưa mua / sản phẩm không tồn tại / khách không tồn tại):
   đáp án đúng là "không tìm thấy" -> đo được hallucination / false-positive.
4. Truy vấn đa dạng: tên chuẩn, một phần, đổi hoa/thường, gõ sai (ngẫu nhiên + phím kề nhau), lỗi khoảng trắng,
   đảo từ, nhiễu, "A hoặc B", paraphrase đồng nghĩa, thuộc tính (màu + loại), câu hội thoại dài.
5. Có nhóm Security (SQL injection, wildcard, ký tự đặc biệt, chuỗi dài, unicode...).
6. Chỉ chọn khách có >= 10 sản phẩm khác nhau (nếu <=5 thì Recall@5 luôn = 100% vì lọc
   theo khách rồi lấy top-5 -> test không có ý nghĩa), chia đều 3 tầng (tier) khách hàng.
7. Toàn bộ dùng random.Random(SEED) -> tái lập 100%.
"""
import json
import os
import random
import re

import pandas as pd

# ----------------------------- CẤU HÌNH ---------------------------------
SEED = 42
N_PER_SUBTYPE = 30          # số case mỗi loại con (tăng lên 50-100 nếu muốn khoảng tin cậy hẹp hơn)
N_SECURITY_PER_FAMILY = 6   # 3 'pure' (chỉ payload) + 3 'appended' (sản phẩm thật + payload)
MIN_PRODUCTS = 10           # khách phải có >= 10 sản phẩm khác nhau
DATA_DIR = os.getenv("B7_DATA_DIR", "data")
IN_PATH = os.path.join(DATA_DIR, "online_retail_clean.csv")
OUT_PATH = os.path.join(DATA_DIR, "test_cases_B7.csv")

TIERS = ["10-30", "31-100", ">100"]  # theo số sản phẩm khác nhau khách đã mua

rng = random.Random(SEED)

# ----------------------------- NẠP DỮ LIỆU ------------------------------
df = pd.read_csv(IN_PATH)
df["desc_norm"] = df["Description"].astype(str).str.strip().str.lower()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
df["InvoiceTotal"] = df.groupby("InvoiceNo")["Revenue"].transform("sum")

MAX_DATE = df["InvoiceDate"].max()
LAST6 = (MAX_DATE - pd.DateOffset(months=6)).normalize()   # "6 tháng gần nhất" tính theo dữ liệu

by_cust = {c: g.reset_index(drop=True) for c, g in df.groupby("CustomerID")}
n_prod = {c: g["desc_norm"].nunique() for c, g in by_cust.items()}
catalog = sorted(df["desc_norm"].unique())


def tier_of(n):
    return "10-30" if n <= 30 else ("31-100" if n <= 100 else ">100")


cust_by_tier = {t: sorted(c for c, n in n_prod.items() if n >= MIN_PRODUCTS and tier_of(n) == t)
                for t in TIERS}
all_valid_customers = sorted(c for t in TIERS for c in cust_by_tier[t])
all_customer_ids = set(by_cust)

# token -> danh sách sản phẩm (dùng cho hard-negative)
tok_index = {}
for d in catalog:
    for t in set(re.findall(r"[a-z]{4,}", d)):
        tok_index.setdefault(t, []).append(d)


# ----------------------------- HÀM HỖ TRỢ -------------------------------
def pick_customer(tier):
    return rng.choice(cust_by_tier[tier])


def pick_row(cust, recent_bias=False):
    g = by_cust[cust]
    if recent_bias and rng.random() < 0.6:
        gr = g[g["InvoiceDate"] >= LAST6]
        if len(gr):
            g = gr.reset_index(drop=True)
    return g.iloc[rng.randrange(len(g))]


def word_re(s):
    return re.compile(r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])")


def gt_for(cust, mask_fn=None, qty_gt=None, inv_gt=None, logic="AND", date_from=None, date_to=None):
    """Tập Description + InvoiceNo thoả mãn điều kiện (khớp ngữ nghĩa SQL ở B7.1)."""
    g = by_cust[cust]
    m = mask_fn(g["desc_norm"]) if mask_fn else pd.Series(True, index=g.index)
    conds = []
    if qty_gt is not None:
        conds.append(g["Quantity"] > qty_gt)
    if inv_gt is not None:
        conds.append(g["InvoiceTotal"] > inv_gt)
    if conds:
        amt = conds[0]
        for c in conds[1:]:
            amt = (amt | c) if logic == "OR" else (amt & c)
        m = m & amt
    if date_from:
        m = m & (g["InvoiceDate"] >= pd.Timestamp(date_from))
    if date_to:
        m = m & (g["InvoiceDate"] < pd.Timestamp(date_to) + pd.Timedelta(days=1))
    sub = g[m]
    return sorted(sub["desc_norm"].unique()), sorted(int(x) for x in sub["InvoiceNo"].unique())


def make_case(ctype, subtype, query, search, cust, gt, inv, tier=None, variant="",
              qty_gt=None, inv_gt=None, logic="AND", date_from=None, date_to=None,
              is_injection=False):
    cust_s = str(cust)
    n = n_prod.get(cust)
    return {
        "Type": ctype, "Subtype": subtype, "Variant": variant,
        "Query": query, "SearchText": search, "CustomerID": cust_s,
        "CustomerTier": tier or (tier_of(n) if n else ""),
        "CustomerNProducts": n if n else "",
        "QtyGt": "" if qty_gt is None else qty_gt,
        "InvoiceTotalGt": "" if inv_gt is None else inv_gt,
        "FilterLogic": logic,
        "DateFrom": date_from or "", "DateTo": date_to or "",
        "ShouldAbstain": len(gt) == 0,
        "ExpectedDescriptions": json.dumps(gt[:50], ensure_ascii=False),
        "ExpectedInvoiceNos": json.dumps(inv[:50]),
        "ExpectedAnswerText": (f"Customer {cust_s} bought " + " | ".join(gt[:3])) if gt
                              else "No matching purchase (system should abstain)",
        "IsInjection": is_injection,
    }


# --- biến đổi chuỗi -----------------------------------------------------
def make_typo(s, n_edits):
    for _ in range(50):
        ch = list(s)
        for _ in range(n_edits):
            idx = [i for i, c in enumerate(ch) if c.isalpha()]
            i = rng.choice(idx)
            op = rng.choice(["del", "swap", "sub", "dup"])
            if op == "del" and len(ch) > 3:
                del ch[i]
            elif op == "swap" and i + 1 < len(ch) and ch[i + 1].isalpha():
                ch[i], ch[i + 1] = ch[i + 1], ch[i]
            elif op == "sub":
                ch[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
            elif op == "dup":
                ch.insert(i, ch[i])
        out = "".join(ch)
        if out != s:
            return out
    return s + "x"


NOISE_TEMPLATES = [
    "{d} or similar product", "similar to {d}", "something like {d}", "the {d} or some accessory",
    "abc {d} xyz", "random {d} test", "foo {d} bar", "{d} or similar",
]


# ----------------------------- BUILDERS ---------------------------------
def b_exact(tier):
    c = pick_customer(tier); d = pick_row(c)["desc_norm"]
    gt, inv = gt_for(c, lambda s: s == d)
    return make_case("Simple", "exact", f"Did customer {c} purchase the product '{d}'?", d, c, gt, inv, tier)


def b_case(tier):
    c = pick_customer(tier); d = pick_row(c)["desc_norm"]
    s = rng.choice([d.upper(), d.title(), "".join(ch.upper() if rng.random() < .5 else ch for ch in d)])
    gt, inv = gt_for(c, lambda x: x == d)
    return make_case("Simple", "case_variant", f"Did customer {c} purchase '{s}'?", s, c, gt, inv, tier)


def b_partial(tier):
    c = pick_customer(tier); d = pick_row(c)["desc_norm"]
    w = d.split()
    if len(w) < 3:
        return None
    i = rng.randrange(len(w) - 1)
    ph = " ".join(w[i:i + 2])
    r = word_re(ph)
    gt, inv = gt_for(c, lambda x: x.apply(lambda v: bool(r.search(v))))
    return make_case("Simple", "partial_phrase", f"Did customer {c} buy something with '{ph}' in its name?",
                     ph, c, gt, inv, tier)


def b_keyword(tier):
    c = pick_customer(tier); d = pick_row(c)["desc_norm"]
    cdescs = by_cust[c]["desc_norm"].unique()
    toks = [t for t in set(re.findall(r"[a-z]{4,}", d))]
    rng.shuffle(toks)
    for t in toks:
        r = word_re(t)
        k = sum(bool(r.search(x)) for x in cdescs)
        if 2 <= k <= 8:
            gt, inv = gt_for(c, lambda x: x.apply(lambda v: bool(r.search(v))))
            return make_case("Ambiguous", "keyword_generic", f"What '{t}' items did customer {c} buy?",
                             t, c, gt, inv, tier)
    return None


def b_typo(tier):
    c = pick_customer(tier); d = pick_row(c)["desc_norm"]
    n = rng.choice([1, 2]); s = make_typo(d, n)
    gt, inv = gt_for(c, lambda x: x == d)
    return make_case("Ambiguous", "typo", f"Did customer {c} buy '{s}'?", s, c, gt, inv, tier, variant=f"{n}edit")


def b_shuffle(tier):
    c = pick_customer(tier); d = pick_row(c)["desc_norm"]
    w = d.split()
    if len(set(w)) < 3:
        return None
    for _ in range(20):
        w2 = w[:]; rng.shuffle(w2)
        if w2 != w:
            break
    s = " ".join(w2)
    if s == d:
        return None
    gt, inv = gt_for(c, lambda x: x == d)
    return make_case("Ambiguous", "word_shuffle", f"Did customer {c} buy '{s}'?", s, c, gt, inv, tier)


def b_noisy(tier):
    c = pick_customer(tier); d = pick_row(c)["desc_norm"]
    s = rng.choice(NOISE_TEMPLATES).format(d=d)
    gt, inv = gt_for(c, lambda x: x == d)
    q = f"Find an invoice with description '{s}' for customer {c}. It could be related to '{d}' or some accessory."
    return make_case("Ambiguous", "noisy_query", q, s, c, gt, inv, tier)


def b_multi(tier):
    c = pick_customer(tier)
    ds = list(by_cust[c]["desc_norm"].unique())
    d1, d2 = rng.sample(ds, 2)
    gt, inv = gt_for(c, lambda x: x.isin([d1, d2]))
    return make_case("Ambiguous", "multi_product", f"Did customer {c} buy '{d1}' or '{d2}'?",
                     f"{d1} or {d2}", c, gt, inv, tier)


def pick_threshold(cands, val):
    below = [x for x in cands if x < val]; above = [x for x in cands if x >= val]
    if rng.random() < 0.6 and below:
        return below[-1]
    return above[0] if above else cands[-1]


def b_fqty(tier):
    c = pick_customer(tier); r = pick_row(c); d = r["desc_norm"]
    q = pick_threshold([1, 5, 10, 24, 50, 100], r["Quantity"])
    gt, inv = gt_for(c, lambda x: x == d, qty_gt=q)
    return make_case("Complex", "filter_qty", f"Did customer {c} buy '{d}' with quantity > {q} in one line?",
                     d, c, gt, inv, tier, qty_gt=q)


def b_fval(tier):
    c = pick_customer(tier); r = pick_row(c); d = r["desc_norm"]
    v = pick_threshold([50, 100, 250, 500, 1000], r["InvoiceTotal"])
    gt, inv = gt_for(c, lambda x: x == d, inv_gt=v)
    return make_case("Complex", "filter_invoice_value",
                     f"Is there an invoice of customer {c} containing '{d}' with total order value > {v}?",
                     d, c, gt, inv, tier, inv_gt=v)


def b_fdate(tier):
    c = pick_customer(tier); r = pick_row(c); d = r["desc_norm"]
    t0 = r["InvoiceDate"].normalize()
    if rng.random() < 0.6:   # cửa sổ có chứa ngày mua
        a, b = t0 - pd.Timedelta(days=rng.randint(10, 90)), t0 + pd.Timedelta(days=rng.randint(10, 90))
    else:                    # cửa sổ lệch đi
        a = t0 + pd.Timedelta(days=rng.randint(30, 60)); b = a + pd.Timedelta(days=rng.randint(30, 90))
    a, b = a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")
    gt, inv = gt_for(c, lambda x: x == d, date_from=a, date_to=b)
    return make_case("Complex", "filter_date_range",
                     f"Did customer {c} buy '{d}' between {a} and {b}?", d, c, gt, inv, tier,
                     date_from=a, date_to=b)


def b_last6(tier):
    c = pick_customer(tier); r = pick_row(c, recent_bias=True); d = r["desc_norm"]
    a = LAST6.strftime("%Y-%m-%d")
    gt, inv = gt_for(c, lambda x: x == d, date_from=a)
    return make_case("Complex", "filter_last_6_months",
                     f"Did customer {c} buy '{d}' in the last 6 months?", d, c, gt, inv, tier, date_from=a)


def b_combo(tier):
    c = pick_customer(tier); r = pick_row(c, recent_bias=True); d = r["desc_norm"]
    a = LAST6.strftime("%Y-%m-%d")
    gt, inv = gt_for(c, lambda x: x == d, qty_gt=5, inv_gt=100, logic="OR", date_from=a)
    q = (f"In the last 6 months, is there any invoice of customer {c} containing '{d}' "
         f"with quantity > 5 or total order value > 100?")
    return make_case("Complex", "filter_combo", q, d, c, gt, inv, tier,
                     qty_gt=5, inv_gt=100, logic="OR", date_from=a)


def b_neg_random(tier):
    c = pick_customer(tier); bought = set(by_cust[c]["desc_norm"])
    d = rng.choice(catalog)
    if d in bought:
        return None
    return make_case("Negative", "neg_not_bought_random", f"Did customer {c} purchase '{d}'?", d, c, [], [], tier)


def b_neg_near(tier):
    """Hard negative: sản phẩm chưa mua nhưng chung từ khoá với sản phẩm khách đã mua."""
    c = pick_customer(tier); bought = set(by_cust[c]["desc_norm"])
    d0 = rng.choice(sorted(bought))
    toks = list(set(re.findall(r"[a-z]{4,}", d0))); rng.shuffle(toks)
    for t in toks:
        cands = [x for x in tok_index.get(t, []) if x not in bought]
        if cands:
            d = rng.choice(cands)
            return make_case("Negative", "neg_not_bought_similar", f"Did customer {c} purchase '{d}'?",
                             d, c, [], [], tier)
    return None



# ---------------- NHÓM MỚI: paraphrase / thuộc tính / hội thoại / lỗi gõ thực tế ----------------
SYNONYMS = {
    "mug": ["cup"], "cup": ["mug"], "box": ["case"], "boxes": ["cases"], "decoration": ["ornament"],
    "light": ["lamp"], "lights": ["lamps"], "necklace": ["pendant"], "sign": ["plaque"], "holder": ["stand"],
    "large": ["big"], "small": ["little"], "mini": ["tiny"], "vintage": ["retro", "antique"], "retro": ["vintage"],
    "christmas": ["xmas"], "gift": ["present"], "bowl": ["dish"], "cushion": ["pillow"], "bottle": ["flask"],
    "purse": ["wallet"], "doormat": ["door mat"], "heart": ["love"], "hanging": ["dangling"], "wooden": ["wood"],
    "bracelet": ["bangle"], "notebook": ["journal"], "umbrella": ["parasol"], "tray": ["platter"],
    "basket": ["hamper"], "jar": ["pot"], "tin": ["can"], "polkadot": ["polka dot", "spotty"],
    "spotty": ["polkadot"], "pack": ["bundle"], "bag": ["tote"], "candle": ["taper"], "garland": ["bunting"],
}
COLORS = ["pink", "blue", "red", "white", "black", "green", "silver", "purple", "cream", "ivory", "gold", "yellow"]
NOUNS = ["bag", "candle", "box", "mug", "bowl", "necklace", "bracelet", "light", "lights", "card", "cushion",
         "frame", "bottle", "holder", "heart", "flower", "glass", "metal", "ceramic", "cake", "paper", "tree",
         "cover", "earrings", "jar", "tray", "sign", "clock", "mirror", "purse"]
KEY_NEIGHBORS = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok", "o": "ipl",
    "p": "ol", "a": "qsz", "s": "awdx", "d": "serf", "f": "drgc", "g": "fthv", "h": "gyjb", "j": "hukn",
    "k": "jilm", "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}
CONVERSATIONAL = [
    "I think she ordered something like {p} a while back, can you check?",
    "my customer says she bought {p} last time, is that right?",
    "was there any {p} in her orders?",
    "can you look up whether this account has {p}? thanks",
    "{p} - did they get it?",
    "quick question, did they ever purchase {p} or was that a different customer",
    "she mentioned {p}, I can't remember if it was on an invoice",
]


def b_synonym(tier):
    c = pick_customer(tier); d = pick_row(c)["desc_norm"]
    w = d.split()
    idx = [i for i, t in enumerate(w) if t in SYNONYMS]
    if not idx or len(w) < 3:
        return None
    rng.shuffle(idx)
    k = rng.choice([1, 2]) if len(idx) > 1 else 1
    for i in idx[:k]:
        w[i] = rng.choice(SYNONYMS[w[i]])
    s = " ".join(w)
    if s == d:
        return None
    gt, inv = gt_for(c, lambda x: x == d)
    return make_case("Semantic", "paraphrase_synonym", f"Did customer {c} buy '{s}'?", s, c, gt, inv, tier,
                     variant=f"{k}word")


def b_attribute(tier):
    c = pick_customer(tier); d = pick_row(c)["desc_norm"]
    toks = set(re.findall(r"[a-z]+", d))
    cols = [x for x in COLORS if x in toks]; nouns = [x for x in NOUNS if x in toks]
    if not cols or not nouns:
        return None
    col, noun = rng.choice(cols), rng.choice(nouns)
    rc, rn = word_re(col), word_re(noun)
    gt, inv = gt_for(c, lambda x: x.apply(lambda v: bool(rc.search(v)) and bool(rn.search(v))))
    if not (1 <= len(gt) <= 10):
        return None
    s = rng.choice([f"{col} {noun}", f"something {col} like a {noun}", f"a {noun} in {col}"])
    return make_case("Semantic", "attribute_color_noun", f"Which {col} {noun} items did customer {c} buy?",
                     s, c, gt, inv, tier)


def b_conversational(tier):
    c = pick_customer(tier); d = pick_row(c)["desc_norm"]
    w = d.split()
    if len(w) >= 3 and rng.random() < 0.5:
        i = rng.randrange(len(w) - 1); p = " ".join(w[i:i + 2]); r = word_re(p)
        gt, inv = gt_for(c, lambda x: x.apply(lambda v: bool(r.search(v)))); var = "phrase"
    else:
        p = d; gt, inv = gt_for(c, lambda x: x == d); var = "full_name"
    s = rng.choice(CONVERSATIONAL).format(p=p)
    return make_case("Semantic", "conversational", f"[customer {c}] {s}", s, c, gt, inv, tier, variant=var)


def b_typo_keyboard(tier):
    c = pick_customer(tier); d = pick_row(c)["desc_norm"]
    n = rng.choice([1, 2])
    for _ in range(30):
        ch = list(d)
        idx = [i for i, x in enumerate(ch) if x in KEY_NEIGHBORS]
        for i in rng.sample(idx, min(n, len(idx))):
            ch[i] = rng.choice(KEY_NEIGHBORS[ch[i]])
        s = "".join(ch)
        if s != d:
            break
    gt, inv = gt_for(c, lambda x: x == d)
    return make_case("Ambiguous", "typo_keyboard", f"Did customer {c} buy '{s}'?", s, c, gt, inv, tier,
                     variant=f"{n}key")


def b_spacing(tier):
    c = pick_customer(tier); d = pick_row(c)["desc_norm"]
    w = d.split()
    mode = rng.choice(["join", "split", "nopunct"])
    if mode == "join" and len(w) >= 3:
        i = rng.randrange(len(w) - 1); w = w[:i] + [w[i] + w[i + 1]] + w[i + 2:]; s = " ".join(w)
    elif mode == "split":
        long_i = [i for i, t in enumerate(w) if len(t) >= 6 and t.isalpha()]
        if not long_i:
            return None
        i = rng.choice(long_i); k = rng.randint(2, len(w[i]) - 2)
        w = w[:i] + [w[i][:k], w[i][k:]] + w[i + 1:]; s = " ".join(w)
    else:
        s = re.sub(r"[^a-z0-9 ]", "", d)
    if s == d:
        return None
    gt, inv = gt_for(c, lambda x: x == d)
    return make_case("Ambiguous", "spacing_noise", f"Did customer {c} buy '{s}'?", s, c, gt, inv, tier, variant=mode)


OFF_CATALOG = [
    "laptop charger usb-c 65w", "iphone 15 pro max case", "garden hose 50 meters", "nike air max sneakers",
    "organic basmati rice 5kg", "quantum flux capacitor", "xqzv plorf wibble", "mens leather wallet brown",
    "gaming mouse rgb wireless", "yoga mat 6mm non slip", "toyota corolla brake pads", "samsung galaxy tablet",
    "baby formula stage 2", "dog food chicken 10kg", "bluetooth noise cancelling headphones", "asdfghjkl",
]


def b_neg_offcat(tier):
    c = pick_customer(tier); d = OFF_CATALOG[len(seen_offcat) % len(OFF_CATALOG)]; seen_offcat.append(d)
    return make_case("Negative", "neg_nonexistent_product", f"Did customer {c} purchase '{d}'?", d, c, [], [], tier)


seen_offcat = []
FAKE_CUSTOMERS = ["0", "99999", "-1", "1234567", "12345", "18288", "11111", "20000"]


def b_neg_cust(tier):
    c = FAKE_CUSTOMERS[len(seen_fake) % len(FAKE_CUSTOMERS)]; seen_fake.append(c)
    if int(c) in all_customer_ids:
        return None
    d = rng.choice(catalog)
    return make_case("Negative", "neg_unknown_customer", f"Did customer {c} purchase '{d}'?", d, c, [], [], "")


seen_fake = []

BUILDERS = [  # (builder, số case)
    b_exact, b_case, b_partial,
    b_keyword, b_typo, b_shuffle, b_noisy, b_multi, b_typo_keyboard, b_spacing,
    b_synonym, b_attribute, b_conversational,
    b_fqty, b_fval, b_fdate, b_last6, b_combo,
    b_neg_random, b_neg_near, b_neg_offcat, b_neg_cust,
]

# ----------------------------- SECURITY ---------------------------------
INJECTION_FAMILIES = {
    "sql_drop": ["'; DROP TABLE transactions; --", "x'; DROP TABLE transactions CASCADE; --", "'); DROP TABLE transactions;--"],
    "sql_tautology": ["' OR '1'='1", "' OR 1=1 --", "\" OR \"\"=\"", "') OR ('a'='a"],
    "sql_union": ["' UNION SELECT NULL,NULL,NULL --", "' UNION SELECT \"CustomerID\"::text,\"Description\" FROM transactions --",
                  "x' UNION ALL SELECT version(),NULL--"],
    "sql_stacked_write": ["'; UPDATE transactions SET \"Quantity\"=0; --", "'; DELETE FROM transactions; --",
                          "'; INSERT INTO transactions DEFAULT VALUES; --", "'; ALTER TABLE transactions ADD COLUMN pwn text; --"],
    "sql_comment": ["--", "/* x */", "#", "-- -"],
    "like_wildcard": ["%", "%%%", "_%", "_____", "%a%"],
    "special_chars": ["it's", "\\", "100\\% cotton", "\"\"\"", "back\\slash'quote"],
    "long_string": ["A" * 5000, "heart " * 800],
    "unicode": ["心形装饰", "🎄🎄🎄", "ｈｅａｒｔ", "\u202etrap"],
    "blank": ["", " ", "\t\n"],
    "control_chars": ["heart<NUL>x", "<NUL>", "line1\nline2\r\n"],   # <NUL> được thay bằng \x00 lúc chạy
}
CUSTOMER_ID_PAYLOADS = ["{c} OR 1=1", "{c}'; DROP TABLE transactions;--", "' OR '1'='1", "{c}) OR (1=1", "%", "{c} UNION SELECT 1"]


def security_cases():
    out = []
    for fam, payloads in INJECTION_FAMILIES.items():
        for i in range(N_SECURITY_PER_FAMILY):
            tier = TIERS[i % 3]; c = pick_customer(tier); d = pick_row(c)["desc_norm"]
            p = payloads[i % len(payloads)]
            if i % 2 == 0:   # pure: chỉ payload -> hệ thống nên không trả sản phẩm nào
                out.append(make_case("Security", fam, f"[payload only] customer {c}: {p[:40]!r}", p, c, [], [], tier,
                                     variant="pure", is_injection=True))
            else:            # appended: sản phẩm thật + payload
                gt, inv = gt_for(c, lambda x: x == d)
                out.append(make_case("Security", fam, f"[product+payload] customer {c}: {d!r} + {p[:30]!r}",
                                     d + " " + p if fam != "blank" else d, c, gt, inv, tier,
                                     variant="appended", is_injection=True))
    for i in range(N_SECURITY_PER_FAMILY):   # injection qua trường CustomerID
        tier = TIERS[i % 3]; c = pick_customer(tier); d = pick_row(c)["desc_norm"]
        p = CUSTOMER_ID_PAYLOADS[i % len(CUSTOMER_ID_PAYLOADS)].format(c=c)
        out.append(make_case("Security", "customer_id_injection", f"customer id = {p!r}, product '{d}'", d, p, [], [], tier,
                             variant="pure", is_injection=True))
    return out


# ----------------------------- SINH & XUẤT ------------------------------
cases, seen = [], set()
for b in BUILDERS:
    got, attempts = 0, 0
    while got < N_PER_SUBTYPE and attempts < N_PER_SUBTYPE * 200:
        attempts += 1
        case = b(TIERS[got % 3])
        if not case:
            continue
        key = (case["Subtype"], case["CustomerID"], case["SearchText"], case["QtyGt"], case["InvoiceTotalGt"],
               case["DateFrom"], case["DateTo"])
        if key in seen:
            continue
        seen.add(key); cases.append(case); got += 1
    if got < N_PER_SUBTYPE:
        print(f"⚠️  {b.__name__}: chỉ sinh được {got}/{N_PER_SUBTYPE} case")

cases += security_cases()
df_cases = pd.DataFrame(cases)
df_cases.insert(0, "CaseID", [f"C{i+1:04d}" for i in range(len(df_cases))])
df_cases.to_csv(OUT_PATH, index=False)

print(f"🎉 Đã tạo {len(df_cases)} test case -> {OUT_PATH}")
print(df_cases.groupby(["Type", "Subtype"]).agg(N=("CaseID", "count"),
                                                 ShouldAbstain=("ShouldAbstain", "mean")).round(2))
print("\nPhân bố tầng khách hàng:\n", df_cases["CustomerTier"].value_counts())