# Q31: Customer Segmentation Using Clustering
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ecommerce_customers.csv")
X = df[["purchase_history", "browsing_minutes", "age"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=4, n_init=10, random_state=42)
df["segment"] = model.fit_predict(X_scaled)

print("Customer segment sizes:")
print(df["segment"].value_counts().sort_index())

print("\nSegment profile (mean values):")
print(df.groupby("segment")[["purchase_history", "browsing_minutes", "age"]].mean().round(2))

plt.figure(figsize=(7, 5))
plt.scatter(df["purchase_history"], df["browsing_minutes"], c=df["segment"], cmap="viridis", s=40)
plt.title("Customer Segments (Purchase History vs Browsing Minutes)")
plt.xlabel("Purchase History (Rs)")
plt.ylabel("Browsing Minutes")
plt.tight_layout()
plt.show()
