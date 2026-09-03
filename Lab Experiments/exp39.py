# Q39: K-Means Clustering for Purchase Behavior + Visualization
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("purchase_behavior.csv")
X = df[["total_amount", "items_purchased"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=4, n_init=10, random_state=42)
df["cluster"] = model.fit_predict(X_scaled)

print("Cluster sizes:")
print(df["cluster"].value_counts().sort_index())
print("\nCluster profile (mean values):")
print(df.groupby("cluster")[["total_amount", "items_purchased"]].mean().round(2))

plt.figure(figsize=(7, 5))
scatter = plt.scatter(df["total_amount"], df["items_purchased"], c=df["cluster"], cmap="viridis", s=45)
plt.title("Customer Clusters: Total Amount vs Items Purchased")
plt.xlabel("Total Amount Spent")
plt.ylabel("Items Purchased")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()
