# Q40: Soccer Player Analysis
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("soccer_players.csv")

print("Top 5 players by goals scored:")
print(df.nlargest(5, "goals")[["name", "goals"]].to_string(index=False))

print("\nTop 5 players by salary:")
print(df.nlargest(5, "salary")[["name", "salary"]].to_string(index=False))

avg_age = df["age"].mean()
print(f"\nAverage age of players: {avg_age:.1f}")

above_avg = df[df["age"] > avg_age]
print(f"\nPlayers above average age ({len(above_avg)} players):")
print(above_avg[["name", "age"]].to_string(index=False))

position_counts = df["position"].value_counts()
plt.figure(figsize=(7, 5))
plt.bar(position_counts.index, position_counts.values, color="teal")
plt.title("Distribution of Players by Position")
plt.xlabel("Position")
plt.ylabel("Number of Players")
plt.tight_layout()
plt.show()
