import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Đường dẫn file gốc và file kết quả
input_file = os.path.join("data", "online_retail_clean.csv")
output_file = os.path.join("data", "online_retail_clean_result.parquet")

# Load mô hình embedding (384 chiều)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Thiết lập kích thước batch để tránh tràn RAM
chunksize = 10000  # bạn có thể điều chỉnh tùy RAM (5000, 20000,...)

# Danh sách lưu embedding
embeddings_list = []
rows_processed = 0

# Đọc dữ liệu theo từng chunk
for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize)):
    # Tạo chuỗi kết hợp từ InvoiceNo và Description
    chunk['combined_text'] = chunk['InvoiceNo'].astype(str) + " " + chunk['Description'].astype(str)

    # Sinh embedding cho batch hiện tại
    embeddings = model.encode(chunk['combined_text'].tolist(), show_progress_bar=True)

    # Lưu embedding vào list
    embeddings_list.extend(embeddings)

    # Báo tiến độ theo cột mốc
    rows_processed += len(chunk)
    print(f"✅ Mốc {i + 1}: đã xử lý {rows_processed} dòng")

# Chuyển embedding thành numpy array
embeddings_array = np.array(embeddings_list)

# Đọc lại toàn bộ file để gắn embedding
df = pd.read_csv(input_file)
df['embedding'] = embeddings_array.tolist()

# Lưu ra file kết quả với hậu tố _result
df.to_parquet(output_file, index=False)

print(f"🎉 Hoàn tất! File kết quả đã lưu tại: {output_file}")
