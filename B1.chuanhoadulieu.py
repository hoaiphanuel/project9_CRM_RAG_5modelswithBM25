import pandas as pd

# Đọc cả 2 sheet từ file Excel gốc
file_path = "data/online_retail_II.xlsx"  # đổi thành đường dẫn file của anh
sheet1 = pd.read_excel(file_path, sheet_name="Year 2009-2010")
sheet2 = pd.read_excel(file_path, sheet_name="Year 2010-2011")

# Gộp dữ liệu lại
data = pd.concat([sheet1, sheet2], ignore_index=True)

# Chuẩn hóa dữ liệu
# 1. Loại bỏ bản ghi thiếu CustomerID
data = data.dropna(subset=["Customer ID"])

# 2. Loại bỏ số lượng <= 0 (trả hàng hoặc lỗi nhập liệu)
data = data[data["Quantity"] > 0]

# 3. Loại bỏ giá <= 0
data = data[data["Price"] > 0]

# 4. Chuẩn hóa tên sản phẩm: chữ thường, bỏ khoảng trắng dư
data["Description"] = data["Description"].str.strip().str.lower()

# 5. Tạo biến doanh thu
data["Revenue"] = data["Quantity"] * data["Price"]

# 6. Chuyển InvoiceDate về dạng datetime chuẩn
data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])

# 7. Đổi tên cột cho thống nhất
data = data.rename(columns={
    "Invoice": "InvoiceNo",
    "StockCode": "StockCode",
    "Description": "Description",
    "Quantity": "Quantity",
    "InvoiceDate": "InvoiceDate",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID",
    "Country": "Country"
})

# Xuất ra file CSV sạch
data.to_csv("data/online_retail_clean.csv", index=False)
print("Đã chuẩn hóa và lưu thành công vào file online_retail_clean.csv")
