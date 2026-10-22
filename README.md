# RAG–CRM retrieval evaluation (UCI Online Retail II)

Data from https://www.kaggle.com/datasets/nathaniel/uci-online-retail-ii-data-set

Run `B1.chuanhoadulieu.py` to generate `online_retail_clean.csv`.
Because of its size, the data files currently in this repository are only illustrative samples.

## Requirements
Python 3.10+, PostgreSQL with the `pgvector` extension, `pip install -r requirements.txt`.
Database settings are read from `PGDATABASE` (default `retaildb`), `PGUSER`, `PGPASSWORD`, `PGHOST`, `PGPORT`.

## Main evaluation (B8) – run in this order
| Step | Script | Output (folder `data/`) |
|---|---|---|
| 1 | `B1.chuanhoadulieu.py` | `online_retail_clean.csv` |
| 2 | `B2.*`, `B3.*` | load data into PostgreSQL, add the embedding columns (MiniLM, BGE, E5) |
| 3 | `B8_0_taotestcase_500.py` | `test_cases_B8.csv` – 500 test queries (90 must be refused) + 120 development queries, fixed seed |
| 4 | `B8_1_Experiment_BM25.py` | `test_results_B8.csv`, `summary_B8.csv`, `tau_B8.csv` – eight methods (substring, BM25, MiniLM, BGE, E5, three BM25+embedding hybrids); refusal thresholds chosen on the development split only |
| 5 | `B8_2_thongke_BM25.py` | `ci_B8.csv` (bootstrap 95% CIs), `pairwise_B8.csv` (exact McNemar / Wilcoxon, Holm-corrected) |
| 6 | `B8_3_Plotdiagram_BM25.py` | figures in `data/plots/` |

Ground truth is always computed from the raw transactions, never from system output.

## Extension (B9) – run after B8
| Script | Purpose | Output |
|---|---|---|
| `B9_0_extra_cases.py` | 386 extra test queries: paraphrases, natural-language questions, short/keyword, category-level, hard negatives. Paraphrases drafted with Qwen2.5-7B (Ollama) and kept after manual review | `test_cases_B9_extra.csv`, `paraphrase_review_full.csv` |
| `B9_1_Experiment.py` | runs the eight methods on core + extra queries, logs per-query distances | `results_B9.csv` |
| `B9_2_Threshold.py` | offline threshold policies (analysis only; the paper reports the frozen B8 thresholds) | `threshold_*_B9.csv` |
| `B9_4_Security.py` | SQL-injection trials (22 payloads x methods), row-level security, overhead, TLS check | `security_B9.csv` |
| `B9_5_ThresholdFree.py` | AUROC, risk–coverage, controlled-risk policy P4 | `threshold_free_B9.csv`, `p4_policy_B9.csv`, `risk_coverage_free_B9.png` |

`B9_3_Generation.py` is an exploratory generation check and is not part of the reported results.

## Where the paper's results come from
- Core retrieval, refusal, latency, statistics (Sec. IV.A–C, IV.E): `test_results_B8.csv`, `summary_B8.csv`, `ci_B8.csv`, `pairwise_B8.csv`, `tau_B8.csv`
- Threshold-free analysis and P4 (Sec. IV.C): `threshold_free_B9.csv`, `p4_policy_B9.csv`
- Security (Sec. IV.D): `security_B9.csv`
- Extended query sets, Table V (Sec. IV.F): `results_B9.csv` scored with the thresholds in `tau_B8.csv`

Files from the original submission (e.g. `B6.*`, `experiment_results.csv`, `test_results.xlsx`, `test_summary.xlsx`) are kept only to document the earlier version; they are superseded by B8/B9.
