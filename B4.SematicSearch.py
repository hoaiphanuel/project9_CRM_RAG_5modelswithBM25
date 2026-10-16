import psycopg2
from sentence_transformers import SentenceTransformer

# Kết nối tới PostgreSQL
conn = psycopg2.connect(
    dbname="retaildb",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# Load mô hình embedding
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Ví dụ mô tả cần tìm
query_text = "white ceramic mug with handle"

# Encode mô tả thành vector
query_embedding = model.encode([query_text])[0]

# Chuyển vector thành chuỗi dạng pgvector '[...]'
embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

# Truy vấn top 5 sản phẩm gần giống nhất
cur.execute("""
    SELECT "InvoiceNo", "Description", "StockCode", "Country"
    FROM transactions
    ORDER BY embedding <-> %s::vector
    LIMIT 5;
""", (embedding_str,))

results = cur.fetchall()

for r in results:
    print(r)

cur.close()
conn.close()
