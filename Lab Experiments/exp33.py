# Q33: Linear Regression for Car Price Prediction (multi-feature)
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("car_price_multi.csv")
X = df[["engine_size", "horsepower", "fuel_efficiency_kmpl"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Model coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.2f}")
print("Intercept:", round(model.intercept_, 2))
print("R-squared on test data:", round(r2_score(y_test, y_pred), 3))

most_influential = X.columns[abs(model.coef_).argmax()]
print(f"\nMost influential factor on price: {most_influential}")
