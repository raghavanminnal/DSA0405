# Q36: Stock Price Variability Analysis
import pandas as pd
import numpy as np

df = pd.read_csv("stock_prices.csv")
prices = df["closing_price"]

mean_price = prices.mean()
std_price = prices.std()
min_price = prices.min()
max_price = prices.max()
price_range = max_price - min_price
cv = (std_price / mean_price) * 100

daily_returns = prices.pct_change().dropna()

print("Stock Price Variability Analysis")
print("Mean closing price:", round(mean_price, 2))
print("Standard deviation:", round(std_price, 2))
print("Minimum price:", round(min_price, 2))
print("Maximum price:", round(max_price, 2))
print("Price range:", round(price_range, 2))
print("Coefficient of variation (%):", round(cv, 2))
print("Average daily return (%):", round(daily_returns.mean() * 100, 3))
print("Daily return volatility (std, %):", round(daily_returns.std() * 100, 3))

if cv < 10:
    print("\nInsight: The stock shows low variability - relatively stable price movement.")
elif cv < 25:
    print("\nInsight: The stock shows moderate variability.")
else:
    print("\nInsight: The stock shows high variability - a relatively volatile price history.")
