import pandas as pd
from sqlalchemy import create_engine
# 1. Đọc dữ liệu từ CSV (đã chuẩn hóa từ 2 sheet Excel)
data = pd.read_csv("data/online_retail_clean.csv")

# 2. Kết nối tới PostgreSQL
# Đổi username, password, host, port theo cấu hình của anh
engine = create_engine("postgresql://postgres:postgres@localhost:5432/retaildb")

# 3. Import dữ liệu vào bảng 'transactions'
data.to_sql("transactions", engine, if_exists="append", index=False)

print("Import thành công!")
