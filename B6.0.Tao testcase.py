import pandas as pd
import random

# Load dữ liệu gốc
df = pd.read_csv("data/online_retail_clean.csv")

# 1. Simple Queries (InvoiceNo)
invoice_samples = random.sample(list(df["InvoiceNo"].dropna().unique()), 50)
simple_cases = []
for i, inv in enumerate(invoice_samples, 1):
    count = df[df["InvoiceNo"] == inv].shape[0]
    simple_cases.append([f"Q{i}", "Simple", inv, count])
df_simple = pd.DataFrame(simple_cases, columns=["QueryID","QueryType","Input (InvoiceNo)","GroundTruth (ExpectedCount)"])

# 2. Complex Queries (CustomerID)
customer_samples = random.sample(list(df["CustomerID"].dropna().unique()), 50)
complex_cases = []
for i, cid in enumerate(customer_samples, 51):
    top_product = df[df["CustomerID"] == cid]["Description"].value_counts().idxmax()
    complex_cases.append([f"Q{i}", "Complex", cid, top_product])
df_complex = pd.DataFrame(complex_cases, columns=["QueryID","QueryType","Input (CustomerID)","GroundTruth (TopProduct)"])

# 3. Semantic Queries (50 câu query tiếng Anh + nhãn)
semantic_queries = [
    ("Products for Christmas gift wrapping","Christmas, Gift wrap"),
    ("Decorations for tea party","Tea set, Cup, Saucer"),
    ("Kitchen utensils","Kitchenware, Utensils"),
    ("Birthday gifts for kids","Birthday, Gift"),
    ("Halloween decorations","Halloween, Decoration"),
    ("Valentine related products","Valentine, Heart"),
    ("Christmas wall decorations","Christmas, Decoration"),
    ("Classic tea set","Tea set"),
    ("Living room decorative lamp","Lamp, Lighting"),
    ("Small gift box","Gift box"),
    ("Wedding decorations","Wedding, Decoration"),
    ("Outdoor BBQ party items","BBQ, Outdoor"),
    ("Wooden toys for children","Toy, Wooden"),
    ("Dining table decorations","Tableware, Decoration"),
    ("Bathroom accessories","Bathroom, Accessory"),
    ("Garden decorations","Garden, Decoration"),
    ("Summer themed products","Summer"),
    ("Winter themed products","Winter"),
    ("Gifts for mothers","Gift, Mother"),
    ("Gifts for friends","Gift, Friend"),
    ("Birthday party decorations","Birthday, Decoration"),
    ("Music related products","Music"),
    ("Vintage decorations","Vintage"),
    ("Handmade products","Handmade"),
    ("Bedroom decorations","Bedroom, Decoration"),
    ("Living room decorations","Living room, Decoration"),
    ("Afternoon tea party items","Tea party"),
    ("Banquet table decorations","Table, Decoration"),
    ("Baby gifts","Baby, Gift"),
    ("Festival decorations","Festival, Decoration"),
    ("Flower related products","Flower"),
    ("Ceramic decorations","Ceramic"),
    ("Glass decorations","Glass"),
    ("Metal decorations","Metal"),
    ("Fabric decorations","Fabric"),
    ("Paper decorations","Paper"),
    ("Plastic decorations","Plastic"),
    ("Wooden decorations","Wood"),
    ("Door hangings","Door, Hanging"),
    ("Tree hangings","Tree, Hanging"),
    ("Wall hangings","Wall, Hanging"),
    ("Ceiling hangings","Ceiling, Hanging"),
    ("Table hangings","Table, Hanging"),
    ("Chair hangings","Chair, Hanging"),
    ("Bed hangings","Bed, Hanging"),
    ("Lamp hangings","Lamp, Hanging"),
    ("Mirror hangings","Mirror, Hanging"),
    ("Photo frame hangings","Frame, Hanging"),
    ("Clock hangings","Clock, Hanging"),
    ("Bookshelf hangings","Bookshelf, Hanging"),
]
semantic_cases = []
for i, (query, label) in enumerate(semantic_queries, 101):
    semantic_cases.append([f"Q{i}", "Semantic", query, label])
df_semantic = pd.DataFrame(semantic_cases, columns=["QueryID","QueryType","Input (NaturalQuery)","GroundTruth (KeywordLabel)"])

# Xuất ra Excel với 3 sheet
with pd.ExcelWriter("data/test_cases.xlsx") as writer:
    df_simple.to_excel(writer, sheet_name="Simple", index=False)
    df_complex.to_excel(writer, sheet_name="Complex", index=False)
    df_semantic.to_excel(writer, sheet_name="Semantic", index=False)

print(">>> test_cases.xlsx created with 150 queries (50 each type)")
