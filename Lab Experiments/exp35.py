# Q35: K-Means for Retail Customer Segmentation
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("retail_customers.csv")
X = df[["total_spent", "visit_frequency"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=3, n_init=10, random_state=42)
df["segment"] = model.fit_predict(X_scaled)

print("Segment sizes:")
print(df["segment"].value_counts().sort_index())

print("\nSegment profile (mean values):")
print(df.groupby("segment")[["total_spent", "visit_frequency"]].mean().round(2))
