"""
B9.1 - Chay thu nghiem truy xuat cho 8 phuong phap: rule, bm25, mini, bge, e5, hyb_mini, hyb_bge, hyb_e5
tren CA tap dev VA tap test cua bo test B8 (data/test_cases_B8.csv) VA bo test bo sung cua B9_0 (data/test_cases_B9_extra.csv,
neu co), ghi DAC TRUNG tung ca de phan tich nguong ve sau.
Che do "cau hoi": voi cac ca co cot QuestionText (nhom natural_question), chay THEM mot luot trong do bo truy xuat nhan NGUYEN CAU HOI
(da bo ma khach hang va dau nhay) thay vi cum tim da tach; luot nay co CaseID ket thuc bang _Q va Module ket thuc bang _Q.

Khac B8.1: script KHONG chon nguong va KHONG tinh "Accepted". Moi ca ghi: khoang cach hang 1 (Top1Distance), khoang cach
mo ta khac ke tiep (Top2Distance), so token cua cum tim (NTok), cac chi so xep hang (Hit@K, MRR), thoi gian.
Viec chon nguong lam OFFLINE trong B9_2_Threshold.py, chi hoc tren dev, ap len test -> khong ro ri thong tin.

Co CHE DO TIEP TUC (RESUME): neu bi ngat (Ctrl+C / mat dien), chay lai se bo qua cac ca da ghi.
Dau ra: data/results_B9.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crm_retrieval as R

CASES_CSVS = ["data/test_cases_B8.csv", "data/test_cases_B9_extra.csv"]
LOG_CSV = "data/results_B9.csv"
RESUME = True
SPLITS_ORDER = ["dev", "test"]      # chay dev truoc de co du lieu hieu chinh som


COLS = ["CaseID", "Split", "Module", "InputMode", "Type", "SubType", "EvalMode", "Model", "NumExpected", "NTok", "NChar",
        "EncodeTime", "DBTime", "ResponseTime", "Top1Distance", "Top2Distance", "Top1BM", "Top1Emb", "Hit@1", "Hit@3", "Hit@5",
        "Hit@10", "MRR", "Top5Invoices", "Top1Text"]


def save_rows(rows):
    pd.DataFrame(rows).reindex(columns=COLS).to_csv(LOG_CSV, mode="a", header=not os.path.exists(LOG_CSV), index=False)


def load_cases():
    parts = []
    for p in CASES_CSVS:
        if not os.path.exists(p):
            print(f"   (khong co {p}, bo qua)")
            continue
        d = pd.read_csv(p, dtype=str, keep_default_na=False)
        if "Split" not in d.columns:
            d["Split"] = "test"
        if "Module" not in d.columns:
            d["Module"] = "core"
        if "QuestionText" not in d.columns:
            d["QuestionText"] = ""
        parts.append(d)
    allc = pd.concat(parts, ignore_index=True)
    assert allc.QueryID.is_unique, "QueryID bi trung giua cac file"
    return allc


def build_rows(case, split, results, truth, text, question):
    out = []
    for name, (rows, t_enc, t_db) in results.items():
        ranked = R.ranked_invoices(rows)
        d1, d2 = R.top_distances(rows)
        row = {"CaseID": case.QueryID + ("_Q" if question else ""), "Split": split,
               "Module": case.Module + ("_Q" if question else ""), "InputMode": "question" if question else "phrase",
               "Type": case.Type, "SubType": case.SubType, "EvalMode": case.EvalMode,
               "Model": name, "NumExpected": len(truth), "NTok": len(R.tokenize(text)), "NChar": len(text),
               "EncodeTime": t_enc, "DBTime": t_db, "ResponseTime": t_enc + t_db, "Top1Distance": d1, "Top2Distance": d2,
               "Top5Invoices": "|".join(ranked[:5]), "Top1Text": (rows[0]["Description"] if rows else "")[:60]}
        if name.startswith("hyb_") and rows:
            row["Top1BM"], row["Top1Emb"] = rows[0]["bm"], rows[0]["emb"]
        if name == "bm25" and not rows:
            row["Top1Distance"] = row["Top2Distance"] = 1.0           # khong khop token nao
        if name.startswith("hyb_") and not rows:
            row["Top1Distance"] = row["Top2Distance"] = row["Top1BM"] = row["Top1Emb"] = 1.0
        if case.EvalMode == "retrieval":
            row.update(R.score_ranking(ranked, truth))
        out.append(row)
    return out


def main():
    print(">>> [1] Nap test case...")
    allc = load_cases()
    print(allc.groupby(["Split", "Module", "EvalMode"]).size().to_string())

    done = set()
    if RESUME and os.path.exists(LOG_CSV):
        old = pd.read_csv(LOG_CSV, dtype={"CaseID": str})
        cnt = old.groupby(["Split", "CaseID"]).Model.nunique()
        done = {k for k, v in cnt.items() if v >= len(R.ALL_METHODS)}
        print(f"   Tiep tuc: da co {len(done)} luot day du trong {LOG_CSV}")
    elif os.path.exists(LOG_CSV):
        os.remove(LOG_CSV)

    print(">>> [2] Ket noi PostgreSQL, nap mo hinh, dung chi so BM25...")
    ret = R.Retriever(load_models=True, need_bm25=True)
    print(">>> [3] Khoi dong may (khong tinh gio)...")
    ret.run_all("warm up", None)

    print("\n>>> [4] Chay...")
    t_start = time.time()
    cum = {m: [] for m in R.ALL_METHODS}
    try:
        for split in SPLITS_ORDER:
            cases = allc[allc.Split == split].reset_index(drop=True)
            for idx, c in cases.iterrows():
                cust = c.CustomerID or None
                truth = set(filter(None, c.ExpectedInvoiceNos.split("|")))
                units = [(False, c.SearchText)]
                if c.QuestionText:
                    units.append((True, c.QuestionText))
                for question, text in units:
                    cid = c.QueryID + ("_Q" if question else "")
                    if (split, cid) in done:
                        continue
                    results = ret.run_all(text, cust)
                    rows = build_rows(c, split, results, truth, R.clean(text), question)
                    save_rows(rows)
                    if c.EvalMode == "retrieval" and c.Module == "core" and not question:
                        for r in rows:
                            cum[r["Model"]].append(r["Hit@5"])
                    el = (time.time() - t_start) / 60
                    print(f"[{split} {idx + 1}/{len(cases)}] {cid} {c.Module}/{c.SubType} {c.EvalMode} cust={cust or '-'} | {el:.1f} phut"
                          + (" | Hit@5 core tich luy: " + " ".join(f"{m}={np.mean(v):.2f}" for m, v in cum.items() if v) if cum["bm25"] else ""))
    except KeyboardInterrupt:
        print("\n⛔ Dung giua chung. Chay lai de TIEP TUC tu cho da dung.")
    ret.close()

    log = pd.read_csv(LOG_CSV, dtype={"CaseID": str})
    t = log[(log.Split == "test") & (log.EvalMode == "retrieval")]
    print("\n=== Tom tat nhanh (tap test, ca truy xuat) : Hit@5 theo nhom ===")
    print(t.pivot_table(index="Module", columns="Model", values="Hit@5").reindex(columns=R.ALL_METHODS).round(3).to_string())
    print(f"\n🎉 Xong. Log: {LOG_CSV}. Buoc tiep: python B9_2_Threshold.py")


if __name__ == "__main__":
    main()
