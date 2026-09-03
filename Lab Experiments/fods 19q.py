import pandas as pd

# Load sales data
df = pd.read_csv("sales_data.csv")

# Calculate Total Sales
df["Total Sales"] = df["Quantity Sold"] * df["Unit Price"]

# Total sales for each product
product_sales = df.groupby("Product")["Total Sales"].sum()

print("Total Sales for Each Product:")
print(product_sales)

# Calculate 20% profit
df["Profit"] = df["Total Sales"] * 0.20

# Overall profit
overall_profit = df["Profit"].sum()

print("\nOverall Profit:", overall_profit)

# Top 5 most profitable products
product_profit = df.groupby("Product")["Profit"].sum()
top_5_products = product_profit.nlargest(5)

print("\nTop 5 Most Profitable Products:")
print(top_5_products)
