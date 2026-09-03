# Q38: Weather Data Variability Analysis
import pandas as pd

df = pd.read_csv("weather_data.csv")

summary = df.groupby("city")["temperature"].agg(
    mean_temp="mean", std_temp="std", min_temp="min", max_temp="max"
)
summary["temp_range"] = summary["max_temp"] - summary["min_temp"]
summary = summary.round(2)

print("City-wise Temperature Summary:")
print(summary)

highest_range_city = summary["temp_range"].idxmax()
most_consistent_city = summary["std_temp"].idxmin()

print(f"\nCity with the highest temperature range: {highest_range_city} ({summary.loc[highest_range_city, 'temp_range']} degrees C)")
print(f"Most consistent city (lowest std dev): {most_consistent_city} ({summary.loc[most_consistent_city, 'std_temp']} degrees C)")
