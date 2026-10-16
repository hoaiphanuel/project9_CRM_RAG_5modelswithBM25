import pandas as pd
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
import psycopg2

# Kết nối PostgreSQL
engine = create_engine("postgresql://postgres:postgres@localhost:5432/retaildb")

print("🔹 Bắt đầu load dữ liệu từ bảng...")
df = pd.read_sql('SELECT "InvoiceNo", "Description" FROM transactions', engine)
print(f"✅ Đã load {len(df)} dòng dữ liệu.")

# Load model embedding
print("🔹 Bắt đầu load model embedding...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Model đã sẵn sàng.")

# Tạo embedding cho cột Description
print("🔹 Bắt đầu tạo embedding cho Description...")
embeddings = model.encode(df["Description"].tolist(), batch_size=64, show_progress_bar=True)
print("✅ Đã tạo xong embedding.")

# Chuẩn bị dữ liệu để update
records = [(embeddings[i].tolist(), df["InvoiceNo"].iloc[i]) for i in range(len(df))]

# Update vào bảng
print("🔹 Bắt đầu update embedding vào PostgreSQL...")
conn = psycopg2.connect("dbname=retaildb user=postgres password=postgres host=localhost")
cur = conn.cursor()
print("🔹 Kết nối xong PostgreSQL...")
# Update từng dòng và in log
for i, (vector, invoice) in enumerate(records, start=1):
    cur.execute('UPDATE transactions SET embedding = %s WHERE "InvoiceNo" = %s', (vector, invoice))
    if i % 100 == 0 or i == len(records):  # log mỗi 100 dòng hoặc dòng cuối
        print(f"✅ Đã update {i}/{len(records)} dòng")

conn.commit()
cur.close()
conn.close()

print("🎉 Hoàn tất update embedding cho toàn bộ dữ liệu!")
