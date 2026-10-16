import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer

# Kết nối Postgres
print(">>> Đang kết nối tới Postgres retaildb ...")
conn = psycopg2.connect(
    host="localhost",
    database="retaildb",
    user="postgres",
    password="postgres"
)
cur = conn.cursor()
print(">>> Đã kết nối thành công tới Postgres")

# Load model cho RAG
print(">>> Đang load model embedding ...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print(">>> Model đã sẵn sàng")

# Đọc file test cases trong thư mục data
print(">>> Đang đọc file data/test_cases.xlsx ...")
xls = pd.ExcelFile("data/test_cases.xlsx")
df_simple = pd.read_excel(xls, "Simple")
df_complex = pd.read_excel(xls, "Complex")
df_semantic = pd.read_excel(xls, "Semantic")
print(">>> Đã load 3 sheet: Simple, Complex, Semantic")

results = []

# 1. Test Simple Queries
print(">>> Bắt đầu kiểm nghiệm Simple Queries")
for _, row in df_simple.iterrows():
    inv = str(row["Input (InvoiceNo)"])   # ép kiểu sang string
    gt_count = int(row["GroundTruth (ExpectedCount)"])
    print(f"   - Query {row['QueryID']} với InvoiceNo={inv}")

    # Rule-based
    cur.execute('SELECT COUNT(*) FROM transactions WHERE "InvoiceNo" = %s', (inv,))
    rb_count = cur.fetchone()[0]
    rb_acc = 1 if rb_count == gt_count else 0

    # RAG-based
    query_text = f"order {inv}"
    vec = model.encode([query_text])[0].tolist()
    vec_str = "ARRAY[" + ",".join(str(x) for x in vec) + "]::vector"
    cur.execute(f'SELECT "InvoiceNo" FROM transactions ORDER BY embedding <-> {vec_str} LIMIT 20')
    invs = [str(r[0]) for r in cur.fetchall()]
    rag_acc = 1 if inv in invs else 0

    results.append(["Simple", row["QueryID"], rb_acc, rag_acc])

# 2. Test Complex Queries
print(">>> Bắt đầu kiểm nghiệm Complex Queries")
for _, row in df_complex.iterrows():
    cid = str(row["Input (CustomerID)"])   # ép kiểu sang string
    gt_product = str(row["GroundTruth (TopProduct)"]).lower()
    print(f"   - Query {row['QueryID']} với CustomerID={cid}")

    # Rule-based
    cur.execute("""
        SELECT "Description"
        FROM transactions
        WHERE "CustomerID" = %s
        GROUP BY "Description"
        ORDER BY SUM("Quantity") DESC
        LIMIT 1
    """, (cid,))
    rb_top = cur.fetchone()[0].lower()
    rb_acc = 1 if gt_product in rb_top else 0

    # RAG-based
    query_text = f"shopping behavior customer {cid}"
    vec = model.encode([query_text])[0].tolist()
    vec_str = "ARRAY[" + ",".join(str(x) for x in vec) + "]::vector"
    cur.execute(f'SELECT "Description" FROM transactions ORDER BY embedding <-> {vec_str} LIMIT 20')
    descs = [r[0].lower() for r in cur.fetchall()]
    rag_acc = 1 if any(gt_product in d for d in descs) else 0

    results.append(["Complex", row["QueryID"], rb_acc, rag_acc])

# 3. Test Semantic Queries
print(">>> Bắt đầu kiểm nghiệm Semantic Queries")
for _, row in df_semantic.iterrows():
    query_text = row["Input (NaturalQuery)"]
    gt_labels = [x.strip().lower() for x in str(row["GroundTruth (KeywordLabel)"]).split(",")]
    print(f"   - Query {row['QueryID']} với câu='{query_text}'")

    # Rule-based: không khả dụng
    rb_acc = None

    # RAG-based
    vec = model.encode([query_text])[0].tolist()
    vec_str = "ARRAY[" + ",".join(str(x) for x in vec) + "]::vector"
    cur.execute(f'SELECT "Description" FROM transactions ORDER BY embedding <-> {vec_str} LIMIT 20')
    descs = [r[0].lower() for r in cur.fetchall()]

    # Precision/Recall/F1
    true_pos = sum(any(label in d for d in descs) for label in gt_labels)
    precision = true_pos / len(descs) if descs else 0
    recall = true_pos / len(gt_labels) if gt_labels else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision+recall) > 0 else 0

    results.append(["Semantic", row["QueryID"], rb_acc, f1])

# Xuất kết quả chi tiết
print(">>> Đang ghi file test_results.xlsx ...")
df_results = pd.DataFrame(results, columns=["QueryType","QueryID","Rule-based Accuracy","RAG-based Accuracy"])
df_results.to_excel("test_results.xlsx", index=False)
print(">>> Đã ghi test_results.xlsx")

# Tính trung bình theo loại truy vấn
print(">>> Đang tính trung bình theo loại truy vấn ...")
summary = df_results.groupby("QueryType").agg({
    "Rule-based Accuracy":"mean",
    "RAG-based Accuracy":"mean"
}).reset_index()

# Trung bình toàn hệ thống
overall = pd.DataFrame([{
    "QueryType":"Overall Average",
    "Rule-based Accuracy": df_results["Rule-based Accuracy"].dropna().mean(),
    "RAG-based Accuracy": df_results["RAG-based Accuracy"].mean()
}])

summary = pd.concat([summary, overall], ignore_index=True)
summary.to_excel("test_summary.xlsx", index=False)
print(">>> Đã ghi test_summary.xlsx")

# In ra summary trên console
print("\n===== SUMMARY RESULTS =====")
print(summary)
print(">>> Hoàn tất kiểm nghiệm độ chính xác")
