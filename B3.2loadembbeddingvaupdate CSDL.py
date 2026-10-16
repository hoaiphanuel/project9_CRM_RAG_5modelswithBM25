import numpy as np
import pandas as pd
import psycopg2

print("🚀 Bắt đầu quá trình update embedding vào PostgreSQL...")

# Load lại embedding và mapping
print("🔹 Đang load embedding từ file...")
embeddings = np.load("embeddings.npy")
df = pd.read_csv("invoice_mapping.csv")
print(f"✅ Đã load {len(df)} embedding từ file.")

# Kết nối PostgreSQL
print("🔹 Kết nối tới PostgreSQL...")
conn = psycopg2.connect("dbname=retaildb user=postgres password=postgres host=localhost")
cur = conn.cursor()

# Chuẩn bị dữ liệu
records = [(embeddings[i].tolist(), df["InvoiceNo"].iloc[i]) for i in range(len(df))]

# Update từng dòng và log tiến trình
print("🔹 Bắt đầu update dữ liệu...")
for i, (vector, invoice) in enumerate(records, start=1):
    cur.execute('UPDATE transactions SET embedding = %s WHERE "InvoiceNo" = %s', (vector, invoice))
    if i % 100 == 0 or i == len(records):
        print(f"✅ Đã update {i}/{len(records)} dòng")

conn.commit()
cur.close()
conn.close()

print("🎉 Hoàn tất quá trình update embedding vào PostgreSQL!")
