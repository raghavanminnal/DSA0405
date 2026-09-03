# Q37: Correlation Between Study Time and Exam Scores
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("study_scores.csv")

correlation = df["study_hours"].corr(df["exam_score"])
print("Correlation between study hours and exam scores:", round(correlation, 3))

if correlation > 0.7:
    print("Insight: Strong positive correlation - more study time is strongly associated with higher scores.")
elif correlation > 0.3:
    print("Insight: Moderate positive correlation.")
else:
    print("Insight: Weak or no correlation.")

# Scatter plot with trend line
plt.figure(figsize=(7, 5))
plt.scatter(df["study_hours"], df["exam_score"], color="steelblue", alpha=0.7)
plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.tight_layout()
plt.show()

# Correlation heatmap-style bar (simple visualization)
plt.figure(figsize=(5, 4))
plt.bar(["Correlation"], [correlation], color="darkorange")
plt.ylim(-1, 1)
plt.title("Study Hours - Exam Score Correlation Coefficient")
plt.tight_layout()
plt.show()
