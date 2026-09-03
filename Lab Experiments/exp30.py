# Q30: CART for Car Price Prediction
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("cars_cart.csv")

le_brand = LabelEncoder()
le_engine = LabelEncoder()
df["brand_enc"] = le_brand.fit_transform(df["brand"])
df["engine_enc"] = le_engine.fit_transform(df["engine_type"])

X = df[["mileage", "age_years", "brand_enc", "engine_enc"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=4, random_state=42)
model.fit(X_train, y_train)

print(f"Model R-squared on test data: {model.score(X_test, y_test):.3f}")

print("\nEnter the new car's details:")
mileage = float(input("Mileage: "))
age = int(input("Age (years): "))
brand = input(f"Brand {list(le_brand.classes_)}: ")
engine = input(f"Engine type {list(le_engine.classes_)}: ")

brand_enc = le_brand.transform([brand])[0]
engine_enc = le_engine.transform([engine])[0]

new_car = [[mileage, age, brand_enc, engine_enc]]
predicted_price = model.predict(new_car)[0]

print(f"\nPredicted price: {predicted_price:.2f}")
print("\nDecision path:")
print(export_text(model, feature_names=["mileage", "age_years", "brand_enc", "engine_enc"]))
