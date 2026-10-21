"""Tuỳ chọn: chạy liên tiếp B7_0 -> B7_6 (dừng nếu một bước lỗi). Bạn có thể bỏ qua file này và chạy từng B7_x như README.
    python run_all.py [--regen] [--no-llm] [--from N]"""
import argparse, os, subprocess, sys
ap = argparse.ArgumentParser(); ap.add_argument("--regen", action="store_true"); ap.add_argument("--no-llm", action="store_true")
ap.add_argument("--from", dest="start", type=int, default=0); a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__)); DATA = os.getenv("B7_DATA_DIR", "data")
steps = [(0, "B7_0_taotestcase.py", []), (1, "B7_1_experiment.py", []), (2, "B7_2_analysis_plots.py", ["--no-show"]),
         (3, "B7_3_hybrid.py", ["--no-show"]), (4, "B7_4_rag_prepare.py", []), (5, "B7_5_rag_generate.py", []), (6, "B7_6_rag_sheet.py", [])]
for n, s, extra in steps:
    if n < a.start: continue
    if n == 0 and not a.regen and os.path.exists(os.path.join(DATA, "test_cases_B7.csv")): print("[0] giữ bộ test sẵn có"); continue
    if n in (5, 6) and a.no_llm: print(f"[{n}] bỏ qua (--no-llm)"); continue
    r = subprocess.run([sys.executable, os.path.join(here, s), *extra])
    if r.returncode: sys.exit(f"❌ bước {n} ({s}) lỗi. Sửa rồi chạy: python run_all.py --from {n}")
print("✅ xong. Tiếp: 2 người chấm data/rag_rating_sheet.csv -> python B7_7_rag_score.py -> python B7_8_paper_figures.py")
