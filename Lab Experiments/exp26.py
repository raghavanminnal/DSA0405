# Q26: Linear Regression for Housing Price Prediction
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("housing.csv")
X = df[["area_sqft", "bedrooms", "location_score"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Model R-squared on test data: {model.score(X_test, y_test):.3f}")

print("\nEnter the new house's features:")
area = float(input("Area (sqft): "))
bedrooms = int(input("Number of bedrooms: "))
location_score = int(input("Location score (1-10): "))

new_house = [[area, bedrooms, location_score]]
predicted_price = model.predict(new_house)[0]

print(f"\nPredicted price: {predicted_price:.2f}")
