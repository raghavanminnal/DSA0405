# Q25: Decision Tree for Iris Flower Classification
import warnings; warnings.filterwarnings("ignore")
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X, y = iris.data, iris.target

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

print("Enter the new flower's measurements (cm):")
sepal_length = float(input("Sepal length: "))
sepal_width = float(input("Sepal width: "))
petal_length = float(input("Petal length: "))
petal_width = float(input("Petal width: "))

new_flower = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(new_flower)[0]

print("\nPredicted species:", iris.target_names[prediction])
