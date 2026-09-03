import pandas as pd

# Load customer data
df = pd.read_csv("customer_data.csv")

# Create spending segments
def segment_customer(spending):
    if spending > 50000:
        return "High Spenders"
    elif spending >= 20000:
        return "Medium Spenders"
    else:
        return "Low Spenders"

df["Spending Segment"] = df["Total Spending"].apply(segment_customer)

# Display customers with segments
print("Customer Segments:")
print(df[["Customer ID", "Total Spending", "Spending Segment"]])

# Calculate average age for each segment
average_age = df.groupby("Spending Segment")["Age"].mean()

print("\nAverage Age of Customers in Each Segment:")
print(average_age)
