import pandas as pd
import psycopg2
import os

# Đọc file kết quả (parquet) đã có embedding
input_file = os.path.join("data", "online_retail_clean_result.parquet")
df = pd.read_parquet(input_file)

# Kết nối tới PostgreSQL
conn = psycopg2.connect(
    dbname="retaildb",  # tên database
    user="postgres",  # user
    password="postgres",  # mật khẩu
    host="localhost",  # ví dụ 'localhost' hoặc IP
    port="5432"
)
cur = conn.cursor()

# Insert dữ liệu theo batch
batch_size = 1000
rows = []

for i, row in df.iterrows():
    # Chuyển embedding list thành chuỗi dạng pgvector '[...]'
    embedding_str = "[" + ",".join(str(x) for x in row['embedding']) + "]"

    rows.append((
        row['InvoiceNo'],
        row['StockCode'],
        row['Description'],
        int(row['Quantity']),
        pd.to_datetime(row['InvoiceDate']),
        float(row['UnitPrice']),
        str(row['CustomerID']) if pd.notna(row['CustomerID']) else None,
        row['Country'],
        float(row['Revenue']),
        embedding_str
    ))

    if len(rows) >= batch_size:
        args_str = ",".join(cur.mogrify(
            "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)", r
        ).decode("utf-8") for r in rows)
        cur.execute("INSERT INTO transactions VALUES " + args_str)
        conn.commit()
        print(f"✅ Đã insert {i + 1} dòng")
        rows = []

# Insert phần còn lại
if rows:
    args_str = ",".join(cur.mogrify(
        "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)", r
    ).decode("utf-8") for r in rows)
    cur.execute("INSERT INTO transactions VALUES " + args_str)
    conn.commit()
    print("🎉 Hoàn tất insert toàn bộ dữ liệu")

cur.close()
conn.close()
