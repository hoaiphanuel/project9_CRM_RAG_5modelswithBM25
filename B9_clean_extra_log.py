"""
B9_clean_extra_log.py - DON LOG KET QUA khi bo test bo sung (data/test_cases_B9_extra.csv) da bi sinh lai.

Van de: B9_1 co che do tiep tuc theo ma ca (X0001...). Neu B9_0 duoc chay lai (noi dung moi, cung ma ca) ma log cu van con,
B9_1 se coi cac ma ca do la "da xong" va giu ket qua cua noi dung CU => log khong khop file test hien tai.
Script nay giu lai chi cac dong cua bo loi (Module == core, tuc 500 ca test + 120 ca dev cua B8), sao luu log cu,
va xoa cac file .shard*.csv. Sau do chay lai B9_1 de chay LAI cac nhom bo sung tren file test hien tai.
"""
import glob
import os
import shutil

import pandas as pd

LOG = "data/results_B9.csv"
log = pd.read_csv(LOG, dtype={"CaseID": str}, low_memory=False)
os.makedirs("data/backup_logs", exist_ok=True)
shutil.copy(LOG, "data/backup_logs/results_B9_before_clean.csv")
for p in glob.glob(LOG.replace(".csv", ".shard*.csv")):
    shutil.move(p, os.path.join("data/backup_logs", os.path.basename(p)))
core = log[log.Module.fillna("core") == "core"]
core.to_csv(LOG, index=False)
print(f"Giu {core.CaseID.nunique()} luot cua bo loi (tu {log.CaseID.nunique()} luot); log cu luu o data/backup_logs/")
print("Buoc tiep theo: python B9_1_Experiment.py   (chi chay cac nhom bo sung con thieu)")
