# Q34: KNN for Treatment Outcome Prediction
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("treatment.csv")

le_gender = LabelEncoder()
df["gender_enc"] = le_gender.fit_transform(df["gender"])

le_outcome = LabelEncoder()
df["outcome_enc"] = le_outcome.fit_transform(df["outcome"])  # Bad=0, Good=1

X = df[["age", "gender_enc", "blood_pressure", "cholesterol"]]
y = df["outcome_enc"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("Model Evaluation on Test Set:")
print("Accuracy: ", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall:   ", round(recall_score(y_test, y_pred), 3))
print("F1-Score: ", round(f1_score(y_test, y_pred), 3))

print("\nPredictions on test set (first 10):")
results = pd.DataFrame({
    "Actual": le_outcome.inverse_transform(y_test[:10]),
    "Predicted": le_outcome.inverse_transform(y_pred[:10])
})
print(results.to_string(index=False))
