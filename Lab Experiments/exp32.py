# Q32: Bivariate Analysis + Linear Regression for House Prices
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("housing.csv")

# Bivariate analysis: house size (area) vs price
plt.figure(figsize=(7, 5))
plt.scatter(df["area_sqft"], df["price"], alpha=0.6, color="steelblue")
plt.title("Bivariate Analysis: House Size vs Price")
plt.xlabel("Area (sqft)")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

correlation = df["area_sqft"].corr(df["price"])
print("Correlation between area and price:", round(correlation, 3))

X = df[["area_sqft"]]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Regression equation: Price = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Area")
print("R-squared on test data:", round(r2_score(y_test, y_pred), 3))
print("RMSE on test data:", round(mean_squared_error(y_test, y_pred) ** 0.5, 2))
