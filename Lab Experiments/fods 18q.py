import pandas as pd
import matplotlib.pyplot as plt

# Data
age = [23, 23, 27, 27, 39, 41, 47, 49, 50,
       52, 52, 54, 56, 57, 58, 60, 61, 62]

fat = [9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2,
       34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7]

df = pd.DataFrame({
    "Age": age,
    "%fat": fat
})

# Mean
print("Mean:")
print(df.mean())

# Median
print("\nMedian:")
print(df.median())

# Standard Deviation
print("\nStandard Deviation:")
print(df.std())

# Boxplot
df.boxplot(column=["Age", "%fat"])
plt.title("Boxplot of Age and %fat")
plt.show()

# Scatter plot
plt.scatter(df["Age"], df["%fat"])
plt.title("Age vs %fat")
plt.xlabel("Age")
plt.ylabel("%fat")
plt.show()

# Q-Q plots
import scipy.stats as stats

stats.probplot(df["Age"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Age")
plt.show()

stats.probplot(df["%fat"], dist="norm", plot=plt)
plt.title("Q-Q Plot of %fat")
plt.show()
