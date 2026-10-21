# Quy trình B7 - chạy TỪNG BƯỚC, kiểm tra sau mỗi bước

Chạy theo đúng thứ tự (mỗi file tự in "đọc / ghi / tiếp" khi bắt đầu, và báo lỗi rõ nếu thiếu file đầu vào).
Tất cả file kết quả nằm trong thư mục `data/` (đổi bằng biến môi trường `B7_DATA_DIR`).

    pip install -r requirements.txt

| Bước | Lệnh | Đọc | Ghi |
|---|---|---|---|
| 0 | `python B7_0_taotestcase.py` | online_retail_clean.csv | test_cases_B7.csv (+ .sha256) |
| 1a | `python B7_1a_indexes.py` (chạy 1 lần, TRƯỚC bước 1) | PostgreSQL | chỉ mục trên bảng transactions (không đổi kết quả, chỉ nhanh hơn) |
| 1 | `python B7_1_experiment.py` | test_cases, PostgreSQL | test_results_B7.csv, hybrid_grid_B7.csv, run_manifest_B7.json |
| 2 | `python B7_2_analysis_plots.py` | test_results | figs/*.png, summary_*.csv, pairwise_tests.csv |
| 3 | `python B7_3_hybrid.py` | hybrid_grid, test_results | hybrid_comparison_B7.csv, figs/fig_hybrid.png |
| 4 | `python B7_4_rag_prepare.py` | test_cases, hybrid_comparison, PostgreSQL | rag_contexts_B7.jsonl |
| 5 | `python B7_5_rag_generate.py` | rag_contexts, biến LLM_* | rag_answers_B7.csv |
| 6 | `python B7_6_rag_sheet.py` | rag_answers, rag_contexts | rag_rating_sheet.csv, rag_key_PRIVATE.csv, rag_auto_scores.csv |
| - | 2 người chấm `rag_rating_sheet.csv` → lưu `rag_rater1.csv`, `rag_rater2.csv` | | |
| 7 | `python B7_7_rag_score.py` | rag_rater1/2, rag_key_PRIVATE | rag_final_labels.csv (+ rag_disagreements.csv) |
| 8 | `python B7_8_paper_figures.py` | test_results | figs_paper/*.png (hình cho bài, tiếng Anh) |

`rag_common.py` là thư viện dùng chung cho bước 4-7 (không chạy trực tiếp). `run_all.py` là tuỳ chọn: chạy 0→6 liên tiếp.

## Kiểm tra sau mỗi bước (dừng nếu lệch)
- **B7_0**: 732 ca (Simple 90, Ambiguous 210, Semantic 90, Complex 150, Negative 120, Security 72). Ghi lại mã băm sha256.
- **B7_1a**: in thời gian TRƯỚC/SAU và dòng "DÙNG chỉ mục ✅". Ghi rõ trong bài: độ trễ đo có/không có chỉ mục (lần chạy cũ, không chỉ mục: ~5.5 s/truy vấn).
- **B7_1** (chậm khi chưa có chỉ mục: 2-3 giờ):
  `test_results_B7.csv` = 3660 dòng; `hybrid_grid_B7.csv` = 7260 dòng; không có dòng "hybrid grid error".
  Bảo mật: SecurityViolation = 0; lỗi chưa xử lý chỉ của rule-based ở họ `control_chars` (4 ca).
- **B7_2**: ở ngưỡng cố định, end-to-end gộp phải tái hiện lần chạy trước: Rule-based ≈ 0.595, MiniLM ≈ 0.780, BGE ≈ 0.814, E5 ≈ 0.888, BM25 ≈ 0.848 (lệch ≤ 0.002 là bình thường). Nếu lệch nhiều hơn: kiểm tra lại bộ test/mã băm.
- **B7_3**: `n_test = 330` cho mọi dòng; in ra alpha, tau của hybrid và các p-value McNemar.
- **B7_4**: "sampled 100 cases: answerable=70, unanswerable=30"; alpha/tau in ra phải trùng bước 3.
- **B7_5**: kiểm tra provider/model in ra; đếm câu trả lời = 300; không có "[LLM ERROR]"; đọc thử vài câu. In toàn văn prompt: `python B7_5_rag_generate.py --prompt`.
- **B7_6**: bảng chấm 300 dòng; điều kiện bị giấu (khoá ở `rag_key_PRIVATE.csv` - đừng đưa cho người chấm).
- **B7_7**: kappa; nếu có bất đồng thì điền `rag_adjudicated.csv` (RowID, Final_Correct, Final_Faithful) rồi chạy lại.

## Nguyên tắc
1. Chốt thiết kế trước khi chạy; không đổi seed hay sửa ca sau khi đã thấy kết quả.
2. Chạy một lần, giữ `run_manifest_B7.json`.
3. Ngưỡng/trọng số chỉ chọn trên nửa DEV (CaseID chẵn); báo cáo trên nửa TEST (CaseID lẻ); RAG chỉ lấy ca từ nửa TEST.
4. Báo cáo ba lớp tách riêng: (a) 732 ca, ngưỡng cố định; (b) nửa TEST + hybrid (ngưỡng đã chỉnh); (c) nghiên cứu thử RAG 100 ca.
5. Code chỉ chia sẻ cho phản biện qua kênh của tạp chí, không đưa lên nơi công khai nếu chưa muốn.
