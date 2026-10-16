import pandas as pd

# Load dữ liệu gốc
df = pd.read_csv("data/online_retail_clean.csv")

# Loại bỏ các dòng thiếu thông tin
df = df.dropna(subset=["Description", "CustomerID"])

# Lấy ngẫu nhiên 500 dòng để thử nghiệm
sample_df = df.sample(n=500, random_state=42)

# Xuất ra file CSV
sample_df.to_csv("data/retail_sample_500.csv", index=False)
