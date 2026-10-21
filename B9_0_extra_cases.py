"""
B9.0 - Sinh BO TEST BO SUNG (them vao bo 500 ca cua B8) de kiem nghiem them cac diem bo test cu chua do duoc.

Chay SAU B8_0 (can data/online_retail_clean.csv va file B8_0). Khong sua data/test_cases_B8.csv.
Dau ra: data/test_cases_B9_extra.csv  (+ data/paraphrase_review_sample.csv de ban duyet tay mot mau nho)

Cac nhom (cot Module). Moi nhom tra loi MOT cau hoi khoa hoc rieng:
  M1 paraphrase        Dien dat lai HOAN TOAN (khong dung tu nao cua ten goc) -> embedding co hon BM25 khi khong con trung tu?
                       (can LLM de viet lai; dat LLM_PROVIDER/LLM_MODEL nhu B9_3. Thieu thi nhom nay bi bo qua)
  M2 hard_negative     Ca PHAI TU CHOI khoa: san pham gan giong san pham khach da mua, ten bia nghe hop ly, ngoai linh vuc,
                       ma khach khong ton tai -> do tinh trang tra loi bia o muc kho
  M3 short_keyword     Truy van ngan (2 tu dau) va tu khoa don, ca tra loi duoc lan phai tu choi -> hieu chinh nguong theo do dai
  M4 natural_question  Cau hoi tu nhien da dang hinh thuc (10 kieu); B9_1 chay them o che do "dua nguyen cau hoi cho bo truy xuat"
  M5 category          Suy dien theo NHOM HANG ("thu de nhap tra nong" -> coc/ly): khong co tu trung voi ten san pham
Nhom paraphrase (v2): chi chon san pham "diễn đạt lại được" (ten 2-5 tu thong dung, khong so, khong tu thuong hieu/thiet ke), moi san pham goc chi dung 1 lan;
  LLM giai thich san pham roi viet cum thay the theo vi du mau; ba cong tu dong: (1) khong trung tu noi dung, (2) khong them tu quang cao ("elegant"...),
  (3) cong DINH DANH: LLM phai chon dung san pham goc giua 5 san pham trong lich su CUA CHINH KHACH, 2 lan voi thu tu khac nhau.
  Sau do bat buoc DUYET TAY: mo data/paraphrase_review_full.csv, dien cot Keep (1 = cum xac dinh dung san pham goc, 0 = bo),
  roi chay:  python B9_0_extra_cases.py --apply-review   -> chi giu cac dong Keep=1 (nhan duoc nguoi xac nhan).
Dap an chuan luon tinh tu du lieu goc. Moi nhom co tap test va tap dev (dev chi dung chon nguong).
"""
import os
import random
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

SEED = 42
B8_0_CANDIDATES = ["B8_0_taotestcase_500.py", "B8.0 taotestcase.py", "B8_0_taotestcase.py"]
OUTPUT_CSV = "data/test_cases_B9_extra.csv"
REVIEW_CSV = "data/paraphrase_review_sample.csv"
FULL_REVIEW_CSV = "data/paraphrase_review_full.csv"

PLAN = {   # (test, dev) so ca moi nhom con
    "paraphrase": (90, 30),          # lay du de sau khi duyet tay con >= 60 test / 20 dev
    "near_variant": (40, 12), "invented": (30, 9), "out_of_domain": (14, 6), "unknown_customer": (10, 3),
    "short_name": (15, 30), "keyword": (15, 30), "short_not_bought": (12, 20), "keyword_not_bought": (8, 20),
    "natural": (60, 20), "natural_not_bought": (20, 0),
    "category": (80, 20), "category_not_bought": (20, 10),
}

# ---------------------------------------------------------------- dung lai buoc nap du lieu cua B8_0
here = os.path.dirname(os.path.abspath(__file__))
src_path = next((p for p in B8_0_CANDIDATES if os.path.exists(p)), None) or next(
    (os.path.join(here, p) for p in B8_0_CANDIDATES if os.path.exists(os.path.join(here, p))), None)
if not src_path:
    raise SystemExit(f"Khong tim thay file B8_0 ({B8_0_CANDIDATES}) trong thu muc hien tai.")
code = open(src_path, encoding="utf8").read().replace("\r", "")
a = code.index("# 1. NAP VA CHUAN HOA")
a = code.rfind("# ====", 0, a)
b = code.index("# 3. SIMPLE")
b = code.rfind("# ====", 0, b)
ns = {"__name__": "b8_0_data"}
exec("import os, random, re\nfrom collections import Counter\nimport pandas as pd\nSEED = 42\nINPUT_CSV = 'data/online_retail_clean.csv'\n" + code[a:b], ns)
m0 = re.search(r"pair = P\.drop_duplicates.*?pool_rep = [^\n]*\n", code, re.S)
exec(m0.group(0), ns)
P, P_by_c, P_by_inv = ns["P"], ns["P_by_c"], ns["P_by_inv"]
ALL_DESC, ALL_DESC_SET, WORD_DF = ns["ALL_DESC"], ns["ALL_DESC_SET"], ns["WORD_DF"]
pair, pool_multi, pool_rep = ns["pair"], ns["pool_multi"], ns["pool_rep"]
qt, pair_lines, inv_str, rec, perturb = ns["qt"], ns["pair_lines"], ns["inv_str"], ns["rec"], ns["perturb"]
random.seed(SEED + 7)

STOP = {"the", "of", "and", "a", "an", "for", "with", "in", "on", "to", "set", "pack", "x"}
TOK = re.compile(r"[a-z]{3,}")


def toks(s):
    return {w for w in TOK.findall(s.lower()) if w not in STOP}


CUSTOMERS = sorted(P_by_c)
CUST_SET = set(CUSTOMERS)
DESC_TOK = {d: toks(d) for d in ALL_DESC}
WORD2DESC = defaultdict(list)
for d, ts in DESC_TOK.items():
    for w in ts:
        WORD2DESC[w].append(d)
LOWER_ALL = [d.lower() for d in ALL_DESC]
used = set()


def mk(module, sub, mode, q, ans, inv="", cid="", text="", qtext=""):
    r = rec(module_type[module], sub, mode, q, ans, inv, cid, text)
    r["Module"] = module
    r["QuestionText"] = qtext
    return r


module_type = {"paraphrase": "Semantic", "category": "Semantic", "hard_negative": "Hard", "short_keyword": "Short", "natural_question": "Natural"}


def pick_pair(pool, key_prefix, tries=2000):
    for _ in range(tries):
        r = pool.iloc[random.randrange(len(pool))]
        k = key_prefix + r._key
        if k not in used:
            used.add(k)
            return r
    return None


# ---------------------------------------------------------------- M2 hard negatives
def g_near_variant():
    for _ in range(300):
        r = pick_pair(pool_multi, "nv:")
        if r is None:
            return None
        c, bdesc = r.CustomerID, r.Description
        bought = set(P_by_c[c].Description)
        cand = {d for w in sorted(DESC_TOK[bdesc], key=lambda w: len(WORD2DESC[w]))[:2] for d in WORD2DESC[w]}
        cand = [d for d in cand if d not in bought and d != bdesc and len(DESC_TOK[d] | DESC_TOK[bdesc]) > 0
                and len(DESC_TOK[d] & DESC_TOK[bdesc]) / len(DESC_TOK[d] | DESC_TOK[bdesc]) >= 0.5]
        if cand:
            d = random.choice(cand)
            q = random.choice(["Did customer {c} purchase {d}?", "Has customer {c} ever ordered {d}?", "Is {d} in the orders of customer {c}?"]).format(c=c, d=qt(d))
            return mk("hard_negative", "near_variant", "abstain", q, "NOT_FOUND", cid=c, text=d)
    return None


def g_invented():
    for _ in range(300):
        a_, b_ = random.choice(ALL_DESC), random.choice(ALL_DESC)
        wa, wb = a_.split(), b_.split()
        if len(wa) < 3 or len(wb) < 3:
            continue
        name = " ".join(wa[:2] + wb[-2:]).upper()
        if name in ALL_DESC_SET or any(name.lower() in x for x in LOWER_ALL) or ("inv:" + name) in used:
            continue
        used.add("inv:" + name)
        c = random.choice(CUSTOMERS)
        q = random.choice(["Did customer {c} purchase {d}?", "Find the invoice where customer {c} bought {d}."]).format(c=c, d=qt(name))
        return mk("hard_negative", "invented", "abstain", q, "NOT_FOUND", cid=str(c), text=name)
    return None


OOD = ["weather forecast for tomorrow", "capital city of france", "reset my account password", "flight tickets to paris", "bitcoin price today",
       "laptop charger", "medical appointment booking", "tax return deadline", "pizza delivery near me", "football match results",
       "how to learn guitar", "car insurance quote", "hotel room reservation", "stock market news", "train timetable",
       "movie showtimes tonight", "mortgage interest rate", "visa application", "phone screen repair", "gym membership"]


CUR = {"split": "test"}


def g_ood():
    pool = OOD[:14] if CUR["split"] == "test" else OOD[14:]      # tach cum tu test / dev de khong trung
    cand = [x for x in pool if ("ood:" + x) not in used]
    if not cand:
        return None
    x = random.choice(cand)
    used.add("ood:" + x)
    c = random.choice(CUSTOMERS)
    q = random.choice(["Did customer {c} purchase {d}?", "customer {c}: {d}?"]).format(c=c, d=qt(x))
    return mk("hard_negative", "out_of_domain", "abstain", q, "NOT_FOUND", cid=str(c), text=x)


def g_unknown_customer():
    for _ in range(100):
        c = random.randint(10 ** 5, 10 ** 6)
        if c in CUST_SET:
            continue
        d = random.choice(ALL_DESC)
        q = f"Did customer {c} purchase {qt(d)}?"
        return mk("hard_negative", "unknown_customer", "abstain", q, "NOT_FOUND", cid=str(c), text=d)
    return None


# ---------------------------------------------------------------- M3 short / keyword
def g_short_name():
    r = pick_pair(pool_multi, "sn:")
    if r is None:
        return None
    c, t = r.CustomerID, " ".join(r.Description.split()[:2])
    g = P_by_c[c]
    sub = g[g.Description.str.lower().str.startswith(t.lower())]
    if len(t) < 5 or sub.empty:
        return None
    q = random.choice(["Customer {c}: any {t} items?", "Find {t} in the orders of customer {c}.", "{t} customer {c}"]).format(c=c, t=qt(t))
    return mk("short_keyword", "short_name", "retrieval", q, f"Customer {c} bought products starting with {t}", inv_str(sub), cid=str(c), text=t)


def rarest_word(d):
    ws = [w for w in TOK.findall(d.lower()) if len(w) >= 4 and w not in STOP]
    return min(ws, key=lambda w: WORD_DF.get(w, 10 ** 9)) if ws else None


def g_keyword():
    r = pick_pair(pool_multi, "kw:")
    if r is None:
        return None
    c, w = r.CustomerID, rarest_word(r.Description)
    if not w:
        return None
    g = P_by_c[c]
    sub = g[g.Description.str.lower().str.contains(r"\b" + re.escape(w) + r"\b", regex=True)]
    if sub.empty:
        return None
    q = random.choice(["Customer {c}: anything with {t}?", "{t} orders for customer {c}", "Did customer {c} buy anything {t}?"]).format(c=c, t=qt(w))
    return mk("short_keyword", "keyword", "retrieval", q, f"Customer {c} bought items containing {w}", inv_str(sub), cid=str(c), text=w)


def g_short_not_bought():
    for _ in range(300):
        c = random.choice(CUSTOMERS)
        d = random.choice(ALL_DESC)
        t = " ".join(d.split()[:2])
        if len(d.split()) < 3 or len(t) < 5 or ("snb:" + str(c) + t) in used:
            continue
        g = P_by_c[c]
        if g.Description.str.lower().str.startswith(t.lower()).any():
            continue
        used.add("snb:" + str(c) + t)
        q = random.choice(["Customer {c}: any {t} items?", "Find {t} in the orders of customer {c}."]).format(c=c, t=qt(t))
        return mk("short_keyword", "short_not_bought", "abstain", q, "NOT_FOUND", cid=str(c), text=t)
    return None


def g_keyword_not_bought():
    for _ in range(300):
        c = random.choice(CUSTOMERS)
        d = random.choice(ALL_DESC)
        w = rarest_word(d)
        if not w or ("knb:" + str(c) + w) in used:
            continue
        if P_by_c[c].Description.str.lower().str.contains(r"\b" + re.escape(w) + r"\b", regex=True).any():
            continue
        used.add("knb:" + str(c) + w)
        q = random.choice(["Customer {c}: anything with {t}?", "Did customer {c} buy anything {t}?"]).format(c=c, t=qt(w))
        return mk("short_keyword", "keyword_not_bought", "abstain", q, "NOT_FOUND", cid=str(c), text=w)
    return None


# ---------------------------------------------------------------- M4 cau hoi tu nhien
TEMPL = [
    "hi, could you check if I (customer {c}) ever bought {d}? thanks",
    "customer number {c}. looking for the invoice with {d}",
    "{d} - did I get this before? id {c}",
    "I can't remember if I ordered {d}. My account is {c}. Can you look?",
    "pls find {d} in order history for {c}",
    "Which of my orders included {d}? (customer {c})",
    "quick q: any record of {d} for customer id {c}??",
    "Could you tell me whether customer {c} has purchased {d} at some point in the past year?",
    "Order lookup | customer: {c} | item: {d}",
    "we are customer {c}; did we receive {d} on an earlier order",
]


def question_for(c, d, form=None):
    t = TEMPL[form if form is not None else random.randrange(len(TEMPL))]
    dd = d
    if random.random() < 0.2:
        dd = d.lower()
    if random.random() < 0.15:
        dd = perturb(dd)
    return t.format(c=c, d=dd)


def strip_q(q, c):
    return re.sub(r"\s+", " ", re.sub(rf"\b{c}\b", "", q).replace('"', "").replace("'", "")).strip()


def g_natural():
    r = pick_pair(pair, "nat:")
    if r is None:
        return None
    c, d = r.CustomerID, r.Description
    q = question_for(c, d)
    return mk("natural_question", "natural", "retrieval", q, f"Customer {c} bought {d}", inv_str(pair_lines(c, d)), cid=str(c), text=d, qtext=strip_q(q, c))


def g_natural_nb():
    for _ in range(300):
        c = random.choice(CUSTOMERS)
        d = random.choice(ALL_DESC)
        if d in set(P_by_c[c].Description) or ("nnb:" + str(c) + d) in used:
            continue
        used.add("nnb:" + str(c) + d)
        q = question_for(c, d)
        return mk("natural_question", "natural_not_bought", "abstain", q, "NOT_FOUND", cid=str(c), text=d, qtext=strip_q(q, c))
    return None


# ---------------------------------------------------------------- M5 nhom hang (suy dien)
CATS = {
    "holiday": (r"christmas|xmas|santa|reindeer|advent|snowflake|tree", ["festive trimmings for the end of December", "presents and decor for the winter holidays", "something sparkly for the year-end season"]),
    "drinkware": (r"\bmug|\bcup|teacup|tumbler|flask|bottle|\bglass", ["something to sip a hot drink from", "drinking vessels for the kitchen", "a container to carry a beverage"]),
    "lighting": (r"light|lamp|lantern|candle|tealight|bulb", ["something to give a soft glow in the evening", "a source of ambience for a dark room", "decor that brightens up a corner at night"]),
    "bags": (r"\bbag\b|tote|pouch|purse|rucksack|backpack", ["something to carry belongings while shopping", "an accessory to hold things when travelling", "a carrier for daily essentials"]),
    "stationery": (r"notebook|pencil|\bpen\b|diary|journal|eraser|stationery|sticker|notepad|planner", ["items for writing and note taking", "supplies for the school desk", "something to jot down appointments"]),
    "kitchen": (r"bowl|\bjar\b|\btin\b|plate|cake|baking|cutlery|spoon|\bfork|knife|teapot|\bjug\b|colander|\bpan\b", ["utensils and containers for cooking and serving", "gear for baking at home", "tableware for serving meals"]),
    "soft_furnishing": (r"cushion|blanket|throw|towel|quilt|doormat|bunting", ["soft furnishings for the sofa", "fabric items to make a home cosy", "cosy textile for the living room"]),
    "toys": (r"\btoy|doll|puzzle|\bgame|teddy|skipping|spinning|playhouse", ["playthings for small children", "entertainment for kids", "a gift to amuse a child"]),
    "garden": (r"garden|plant|\bseed|flower|watering|\bbird|\bherb", ["things for the yard and outdoor planting", "supplies for growing greenery", "outdoor gear for plants"]),
    "jewellery": (r"necklace|bracelet|earring|jewel|brooch|charm|\bbead", ["personal adornment worn on the body", "accessories to wear for a special occasion", "small decorative items worn as accessories"]),
    "gift_wrap": (r"wrap|ribbon|gift|\btag\b|\bcard\b|paper", ["materials to present a present nicely", "packaging for giving presents", "something to dress up a parcel"]),
    "storage": (r"\bbox\b|basket|storage|drawer|crate|trunk|chest|cabinet|\bhook|shelf|rack", ["places to keep and organise belongings", "containers to tidy up a room", "furniture bits for organising"]),
}
CAT_NAMES = list(CATS)
cat_cache = {}


def cust_cat_lines(c, cat):
    key = (c, cat)
    if key not in cat_cache:
        g = P_by_c[c]
        cat_cache[key] = g[g.Description.str.lower().str.contains(CATS[cat][0], regex=True)]
    return cat_cache[key]


def g_category():
    for _ in range(500):
        cat = random.choice(CAT_NAMES)
        c = random.choice(CUSTOMERS)
        sub = cust_cat_lines(c, cat)
        phrase = random.choice(CATS[cat][1])
        k = f"cat:{c}|{cat}|{phrase}"
        if sub.empty or k in used:
            continue
        used.add(k)
        q = random.choice(["Did customer {c} buy {p}?", "Find the invoices of customer {c} for {p}.", "customer {c} - anything like {p}?"]).format(c=c, p=qt(phrase))
        return mk("category", "category_" + cat, "retrieval", q, f"Customer {c} bought {cat} items", inv_str(sub), cid=str(c), text=phrase)
    return None


def g_category_nb():
    for _ in range(500):
        cat = random.choice(CAT_NAMES)
        c = random.choice(CUSTOMERS)
        phrase = random.choice(CATS[cat][1])
        k = f"catnb:{c}|{cat}|{phrase}"
        if not cust_cat_lines(c, cat).empty or k in used:
            continue
        used.add(k)
        q = random.choice(["Did customer {c} buy {p}?", "customer {c} - anything like {p}?"]).format(c=c, p=qt(phrase))
        return mk("category", "category_not_bought", "abstain", q, "NOT_FOUND", cid=str(c), text=phrase)
    return None


# ---------------------------------------------------------------- M1 dien dat lai (can LLM)
BRAND = {"retrospot", "spaceboy", "dolly", "regency", "marrakesh", "marakesh", "medina", "babushka", "scotty", "paisley", "pimpernel",
         "bohemian", "jumbo", "dotcomgiftshop", "polkadot", "doily", "doilies", "gingham", "union", "flag", "poppy", "poppies", "scandinavian",
         "victorian", "edwardian", "pantry", "regal", "skull", "skulls", "ladybird", "ladybug", "woodland", "gumball", "love", "building"}
FLUFF = {"elegant", "stylish", "modern", "eco-friendly", "premium", "luxury", "high-quality", "sleek", "trendy", "chic", "contemporary",
         "artisan", "handcrafted", "versatile", "multifunctional", "smart", "ergonomic", "durable", "decorative", "inspired", "compact", "classic"}


def paraphrasable(d):
    ws = d.lower().split()
    if not (2 <= len(ws) <= 5) or any(not re.fullmatch(r"[a-z'\-]+", w) for w in ws):
        return False
    cw = [w for w in ws if w not in STOP]
    if len(cw) < 2 or set(cw) & BRAND:
        return False
    return all(all(len(p) < 4 or WORD_DF.get(p, 0) >= 4 for p in w.replace("-", " ").split()) for w in cw)     # WORD_DF chi dem tu >= 4 chu cai


FEWSHOT = [("blue spotty ceramic egg cup", "small porcelain holder for a boiled egg with dots on it, blue"),
           ("red wooden toy train set", "painted timber locomotive with carriages for children to play with"),
           ("glass cake stand with dome", "raised transparent platter with a cover for displaying desserts"),
           ("white cotton storage bag", "plain fabric sack for keeping things, light colour")]
SYS_PARA = ("You write short search phrases that a customer might use to describe a retail product when they do NOT remember its exact name. "
            "Keep the SAME product type and the SAME attributes (colour, material, shape, size, theme). Never add attributes that are not in the name. "
            "Never use the forbidden words. Reply with the phrase only (4 to 9 words), no quotes, no explanation.")
VARIANTS = ["Describe it using different everyday words.", "Describe what it is and what it is used for.", "Describe its look and material using different words."]


def para_prompt(d, forb, variant):
    ex = "\n".join(f'Product name: {a}\nPhrase: {b}' for a, b in FEWSHOT)
    return f"{ex}\n\nProduct name: {d}\nForbidden words: {', '.join(forb) or '(none)'}\n{VARIANTS[variant]}\nPhrase:"


def identify_ok(llm, p, d, c):
    """Cong dinh danh: LLM phai chon dung san pham goc trong 5 san pham cua chinh khach (2 thu tu khac nhau)."""
    others = [x for x in dict.fromkeys(P_by_c[c].Description) if x != d]
    if len(others) < 4:
        return False
    opts = random.sample(others, 4) + [d]
    for _ in range(2):
        random.shuffle(opts)
        listing = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(opts))
        out = llm.chat("You match a customer's description to a product name.",
                       f'A customer describes a product as: "{p}".\nWhich product name in the list is the best match?\n{listing}\nAnswer with the number only.',
                       max_tokens=8, temperature=0.0)
        m = re.search(r"\d+", out)
        if not m or not (1 <= int(m.group()) <= 5) or opts[int(m.group()) - 1] != d:
            return False
    return True


def make_paraphrase(d, c):
    import llm
    forb = sorted(toks(d))
    for k in range(3):
        out = llm.chat(SYS_PARA, para_prompt(d, forb, k), max_tokens=40, temperature=0.0)
        p = out.strip().strip('"').split("\n")[0].replace("Phrase:", "").strip().strip('"')
        w = toks(p)
        if not (3 <= len(p.split()) <= 11) or (w & toks(d)) or p.lower() == d.lower():
            continue
        if any(x in FLUFF for x in re.findall(r"[a-z\-]+", p.lower()) if x not in d.lower()):
            continue
        if identify_ok(llm, p, d, c):
            return p
    return None


used_orig = set()


def g_paraphrase():
    for _ in range(60):
        r = pick_pair(pool_multi, "pa:")
        if r is None:
            return None
        c, d = r.CustomerID, r.Description
        if d in used_orig or not paraphrasable(d):
            continue
        p = make_paraphrase(d, c)
        if not p:
            continue
        used_orig.add(d)
        q = random.choice(["Did customer {c} buy {p}?", "customer {c}: do they have {p} in any order?"]).format(c=c, p=qt(p))
        res = mk("paraphrase", "paraphrase", "retrieval", q, f"Customer {c} bought {d}", inv_str(pair_lines(c, d)), cid=str(c), text=p)
        res["Original"] = d
        return res
    return None


GEN = {"paraphrase": g_paraphrase, "near_variant": g_near_variant, "invented": g_invented, "out_of_domain": g_ood,
       "unknown_customer": g_unknown_customer, "short_name": g_short_name, "keyword": g_keyword,
       "short_not_bought": g_short_not_bought, "keyword_not_bought": g_keyword_not_bought, "natural": g_natural,
       "natural_not_bought": g_natural_nb, "category": g_category, "category_not_bought": g_category_nb}


def main():
    try:
        import llm                                   # nap llm_config.py (dat khoa/mo hinh trong file, khong can bien moi truong)
    except Exception as e:                           # noqa: BLE001
        print("⚠️  Khong nap duoc llm.py:", e)
    key_ok = os.getenv("LLM_PROVIDER", "anthropic").lower() == "ollama" or any(
        os.getenv(k) and not os.getenv(k).startswith("PASTE") for k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"))
    llm_ok = (bool(os.getenv("LLM_MODEL")) and key_ok) or os.getenv("MOCK_LLM") == "1"
    rows = []
    for name, (n_test, n_dev) in PLAN.items():
        if name == "paraphrase" and not llm_ok:
            print("⚠️  Bo qua nhom 'paraphrase': chua dien LLM_MODEL / khoa API trong llm_config.py (xem llm.py). Chay lai sau khi dat bien moi truong de them nhom nay.")
            continue
        for split, n in (("test", n_test), ("dev", n_dev)):
            CUR["split"] = split
            got = 0
            for _ in range(n * 40):
                if got >= n:
                    break
                r = GEN[name]()
                if r is None or (r["EvalMode"] == "retrieval" and r["NumExpected"] < 1):
                    continue
                r["Split"] = split
                rows.append(r)
                got += 1
            if got < n:
                print(f"⚠️  {name}/{split}: chi tao duoc {got}/{n} ca")
    out = pd.DataFrame(rows).drop_duplicates("Query").reset_index(drop=True)
    out.insert(0, "QueryID", [f"X{i + 1:04d}" for i in range(len(out))])
    cols = ["QueryID", "Split", "Module", "Type", "SubType", "EvalMode", "Query", "QuestionText", "CustomerID", "SearchText",
            "ExpectedInvoiceNos", "NumExpected", "ExpectedAnswerText"]
    out = out[cols + [c for c in out.columns if c not in cols]]
    r_, a_ = out[out.EvalMode == "retrieval"], out[out.EvalMode == "abstain"]
    assert (r_.NumExpected >= 1).all() and (a_.NumExpected == 0).all() and (out.SearchText != "").all()
    os.makedirs("data", exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("\n" + out.groupby(["Split", "Module", "SubType" if False else "EvalMode"]).size().to_string())
    print(f"\nTong: test={int((out.Split == 'test').sum())}, dev={int((out.Split == 'dev').sum())}, phai tu choi (test)={int(((out.Split == 'test') & (out.EvalMode == 'abstain')).sum())}")
    if "Original" in out:
        pr = out[out.Module == "paraphrase"]
        if len(pr):
            pr[["QueryID", "Split", "CustomerID", "Original", "SearchText"]].assign(Keep="", GhiChu="").to_csv(
                FULL_REVIEW_CSV, index=False, encoding="utf-8-sig")
            print(f"Da ghi {len(pr)} cum dien dat lai de DUYET TAY: {FULL_REVIEW_CSV}")
            print("  -> dien cot Keep: 1 = cum xac dinh DUNG san pham goc cua khach, 0 = bo; roi chay: python B9_0_extra_cases.py --apply-review")
    print(f"🎉 Da tao {OUTPUT_CSV}")


def apply_review():
    rev = pd.read_csv(FULL_REVIEW_CSV, dtype=str, keep_default_na=False)
    out = pd.read_csv(OUTPUT_CSV, dtype=str, keep_default_na=False)
    mark = {(r.CustomerID, r.Original): r.Keep.strip() for r in rev.itertuples()}
    is_p = out.Module == "paraphrase"
    keep = out.apply(lambda r: mark.get((r.CustomerID, r.get("Original", "")), "") == "1", axis=1)
    unmarked = int((is_p & ~out.apply(lambda r: (r.CustomerID, r.get("Original", "")) in mark and mark[(r.CustomerID, r.get("Original", ""))] in ("0", "1"), axis=1)).sum())
    if unmarked:
        print(f"⚠️  {unmarked} cum chua dien Keep (0/1); chung se bi BO. Dien du roi chay lai neu muon giu.")
    n_rev = int((rev.Keep.str.strip().isin(["0", "1"])).sum())
    n_ok = int((rev.Keep.str.strip() == "1").sum())
    if n_rev:
        print(f"Ty le cum duoc xac nhan dung: {n_ok}/{n_rev} = {n_ok / n_rev:.0%}  (bao cao con so nay trong bai)")
    new = out[~is_p | keep]
    os.replace(OUTPUT_CSV, OUTPUT_CSV.replace(".csv", "_before_review.csv"))
    new.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(new[new.Module == "paraphrase"].groupby("Split").size().to_dict(), "cum paraphrase con lai (muc tieu >=60 test, >=20 dev)")
    print(f"Da cap nhat {OUTPUT_CSV} (ban truoc review luu thanh *_before_review.csv)")


if __name__ == "__main__":
    if "--apply-review" in sys.argv:
        apply_review()
    else:
        main()
