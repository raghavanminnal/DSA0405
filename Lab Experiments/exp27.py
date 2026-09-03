# Q27: Logistic Regression for Customer Churn Prediction
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("churn.csv")
X = df[["usage_minutes", "contract_duration_months", "support_calls"]]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

print(f"Model accuracy on test data: {model.score(X_test_scaled, y_test):.2f}")

print("\nEnter the new customer's details:")
usage = float(input("Usage minutes: "))
duration = int(input("Contract duration (months): "))
calls = int(input("Number of support calls: "))

new_customer = scaler.transform([[usage, duration, calls]])
prediction = model.predict(new_customer)[0]

print("\nPrediction:", "Customer will churn" if prediction == 1 else "Customer will not churn")
