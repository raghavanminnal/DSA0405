# Q28: K-Means Clustering for Customer Segmentation
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("customers_shopping.csv")
X = df[["annual_spend", "purchase_frequency", "avg_basket_size"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=4, n_init=10, random_state=42)
model.fit(X_scaled)

print("Enter the new customer's shopping features:")
spend = float(input("Annual spend: "))
frequency = float(input("Purchase frequency (times/year): "))
basket = float(input("Average basket size: "))

new_customer = scaler.transform([[spend, frequency, basket]])
segment = model.predict(new_customer)[0]

print(f"\nThis customer belongs to Segment {segment}")
print("Segment sizes:", pd.Series(model.labels_).value_counts().sort_index().to_dict())
