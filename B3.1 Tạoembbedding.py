import pandas as pd
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
import numpy as np

print("🚀 Bắt đầu quá trình tạo embedding...")

# Kết nối PostgreSQL
print("🔹 Kết nối tới PostgreSQL...")
engine = create_engine("postgresql://postgres:postgres@localhost:5432/retaildb")

# Load dữ liệu
print("🔹 Đang load dữ liệu từ bảng...")
df = pd.read_sql('SELECT "InvoiceNo", "Description" FROM transactions', engine)
print(f"✅ Đã load {len(df)} dòng dữ liệu.")

# Load model
print("🔹 Đang load model embedding...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Model đã sẵn sàng.")

# Tạo embedding
print("🔹 Bắt đầu tạo embedding cho Description...")
embeddings = model.encode(df["Description"].tolist(), batch_size=64, show_progress_bar=True)
print("✅ Đã tạo xong embedding.")

# Lưu ra file
print("🔹 Đang lưu embedding và mapping ra file...")
np.save("embeddings.npy", embeddings)
df["InvoiceNo"].to_csv("invoice_mapping.csv", index=False)
print("🎉 Hoàn tất quá trình tạo và lưu embedding!")
