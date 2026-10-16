import pandas as pd
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
import psycopg2

# Kết nối PostgreSQL
engine = create_engine("postgresql://postgres:postgres@localhost:5432/retaildb")

# Load dữ liệu từ bảng (dùng dấu ngoặc kép cho tên cột)
df = pd.read_sql('SELECT "InvoiceNo", "Description" FROM transactions', engine)
print("bắt đầu tranning")

# Load model embedding
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Tạo embedding cho cột Description
embeddings = model.encode(df["Description"].tolist())
print("tranning xong!")

# Chèn embedding vào bảng
conn = psycopg2.connect("dbname=retaildb user=postgres password=postgres host=localhost")
cur = conn.cursor()
print("bat dau truyen!")

for i, row in df.iterrows():
    vector = embeddings[i].tolist()
    print(i)
    cur.execute(
        'UPDATE transactions SET embedding = %s WHERE "InvoiceNo" = %s',
        (vector, row["InvoiceNo"])
    )

conn.commit()
cur.close()
conn.close()

print("Đã cập nhật embedding thành công!")
