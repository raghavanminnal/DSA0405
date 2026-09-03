# Q24: KNN Classifier for medical condition prediction
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("patients.csv")
X = df[["symptom1_fever", "symptom2_fatigue", "symptom3_pain", "symptom4_cough"]]
y = df["condition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k = int(input("Enter the value of k (number of neighbors): "))

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)

print(f"Model accuracy on test data: {model.score(X_test_scaled, y_test):.2f}")

print("\nEnter the new patient's symptom scores (0-10):")
fever = float(input("Fever severity: "))
fatigue = float(input("Fatigue level: "))
pain = float(input("Pain level: "))
cough = float(input("Cough severity: "))

new_patient = scaler.transform([[fever, fatigue, pain, cough]])
prediction = model.predict(new_patient)[0]

print("\nPrediction:", "Has the medical condition" if prediction == 1 else "Does not have the medical condition")
