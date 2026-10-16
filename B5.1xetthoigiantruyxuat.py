import psycopg2
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# [Mốc 1] Kết nối tới Postgres
conn = psycopg2.connect(
    dbname="retaildb",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# [Mốc 2] Load model để tạo embedding cho truy vấn RAG
model = SentenceTransformer("all-MiniLM-L6-v2")

# [Mốc 3] Hàm đo Rule-based
def query_rule_based(sql, params=None, repeat=5):
    times = []
    correct = 0
    for _ in range(repeat):
        start = time.time()
        cur.execute(sql, params)
        rows = cur.fetchall()
        end = time.time()
        times.append(end - start)
        if len(rows) > 0:
            correct += 1
    return np.mean(times), correct/repeat*100, len(rows)

# [Mốc 4] Hàm đo RAG-based (ép kiểu sang vector)
def query_rag(query_text, repeat=5, top_k=5):
    vec = model.encode([query_text])[0].tolist()
    vec_str = "ARRAY[" + ",".join(str(x) for x in vec) + "]::vector"

    times = []
    correct = 0
    results = []
    for _ in range(repeat):
        start = time.time()
        cur.execute(f'SELECT "InvoiceNo" FROM transactions ORDER BY embedding <-> {vec_str} LIMIT %s', (top_k,))
        invs = [row[0] for row in cur.fetchall()]
        cur.execute('SELECT * FROM transactions WHERE "InvoiceNo" = ANY(%s)', (invs,))
        rows = cur.fetchall()
        end = time.time()
        times.append(end - start)
        results.append(len(rows))
        if len(rows) > 0:
            correct += 1
    return np.mean(times), correct/repeat*100, np.mean(results)

# [Mốc 5] Load toàn bộ test case từ file CSV
df_test = pd.read_csv("data/online_retail_test.csv")

invoice_nos = df_test["InvoiceNo"].dropna().unique()
customer_ids = df_test["CustomerID"].dropna().unique()
keywords = df_test["Description"].dropna().unique()[:10]  # lấy 10 từ khóa sản phẩm đầu tiên

tasks = []
for inv in invoice_nos:
    tasks.append(("Truy xuất đơn hàng",
                  'SELECT * FROM transactions WHERE "InvoiceNo" = %s', (str(inv),), f"order {inv}"))
for cid in customer_ids:
    tasks.append(("Phân tích hành vi mua sắm",
                  'SELECT "StockCode", SUM("Quantity") FROM transactions WHERE "CustomerID" = %s GROUP BY "StockCode"', (str(cid),), f"shopping behavior customer {cid}"))
for kw in keywords:
    tasks.append(("Gợi ý sản phẩm cá nhân hóa", None, None, kw))

# [Mốc 6] Chạy thử nghiệm và in kết quả từng dòng
results = []
for task_name, sql, params, rag_query in tasks:
    if sql:
        rb_time, rb_acc, rb_count = query_rule_based(sql, params, repeat=5)
    else:
        rb_time, rb_acc, rb_count = (None, None, None)
    rag_time, rag_acc, rag_count = query_rag(rag_query, repeat=5)

    print(f"\n--- {task_name} ---")
    if sql:
        print(f"Rule-based: {rb_time:.4f} giây, Độ chính xác {rb_acc:.2f}%, Số kết quả {rb_count}")
    else:
        print("Rule-based: Không khả dụng")
    print(f"RAG-based: {rag_time:.4f} giây, Độ chính xác {rag_acc:.2f}%, Số kết quả {rag_count}")

    results.append({
        "Tác vụ": task_name,
        "Rule-based (thời gian)": f"{rb_time:.4f} giây" if rb_time else "Không khả dụng",
        "RAG-based (thời gian)": f"{rag_time:.4f} giây",
        "Rule-based (độ chính xác)": f"{rb_acc:.2f}%" if rb_acc is not None else "Không khả dụng",
        "RAG-based (độ chính xác)": f"{rag_acc:.2f}%",
        "Số kết quả Rule-based": rb_count,
        "Số kết quả RAG-based": rag_count
    })

# [Mốc 7] In bảng tổng quát bằng pandas
df_results = pd.DataFrame(results)
print("\n=== Bảng tổng quát ===")
print(df_results.to_string(index=False))

# [Mốc 8] Lưu file CSV ngay cạnh script
df_results.to_csv("experiment_results.csv", index=False)
print("\n>>> File kết quả đã lưu: experiment_results.csv")

# [Mốc 9] Trung bình toàn hệ thống
avg_rule = np.mean([query_rule_based(t[1], t[2], repeat=5)[0] for t in tasks if t[1]])
avg_rag = np.mean([query_rag(t[3], repeat=5)[0] for t in tasks])
print(f"\n>>> Trung bình toàn hệ thống: Rule-based {avg_rule:.4f} giây, RAG-based {avg_rag:.4f} giây")
