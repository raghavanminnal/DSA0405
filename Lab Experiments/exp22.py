"""
Q22. Customer Review Analysis: Confidence Interval for Mean Rating
-----------------------------------------------------------------------
Scenario: As an analyst for a popular online shopping website, the
task is to analyze customer reviews and provide insights on the
average rating and customer satisfaction level for a specific product
category, using data from "customer_reviews.csv" (contains product
ratings).

This program:
    1. Loads the review ratings using Pandas.
    2. Calculates descriptive statistics (mean rating, std deviation).
    3. Calculates the customer satisfaction level (percentage of
       ratings that are 4 stars or higher).
    4. Uses Pandas together with SciPy to calculate a confidence
       interval to estimate the true population mean rating.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

CSV_FILE = "customer_reviews.csv"
CONFIDENCE_LEVEL = 0.95
SATISFACTION_THRESHOLD = 4   # ratings >= 4 are considered "satisfied"

# ---------------------------------------------------------------
# Create a sample customer_reviews.csv if it does not already exist
# (so the script can be run/tested end-to-end).
# Remove this block once you have your real data file.
# ---------------------------------------------------------------
if not os.path.exists(CSV_FILE):
    rng = np.random.default_rng(11)
    n = 150
    ratings = rng.choice([1, 2, 3, 4, 5], size=n,
                          p=[0.03, 0.07, 0.15, 0.35, 0.40])
    sample = pd.DataFrame({
        "Review ID": range(1, n + 1),
        "Product Category": rng.choice(
            ["Electronics", "Home & Kitchen", "Fashion"], size=n),
        "Rating": ratings
    })
    sample.to_csv(CSV_FILE, index=False)
    print(f"'{CSV_FILE}' not found — a sample file was generated for demo purposes.\n")

# ---------------------------------------------------------------
# 1. Load the data with Pandas
# ---------------------------------------------------------------
df = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df)} customer reviews from '{CSV_FILE}'.")
print(df.head(), "\n")

ratings = df["Rating"]

# ---------------------------------------------------------------
# 2. Descriptive statistics
# ---------------------------------------------------------------
mean_rating = ratings.mean()
std_rating = ratings.std()          # sample standard deviation
n = ratings.count()

print("--- Descriptive Statistics ---")
print(f"Number of reviews:      {n}")
print(f"Average rating:         {mean_rating:.3f}")
print(f"Standard deviation:     {std_rating:.3f}")

# ---------------------------------------------------------------
# 3. Customer satisfaction level
# ---------------------------------------------------------------
satisfied = (ratings >= SATISFACTION_THRESHOLD).sum()
satisfaction_pct = satisfied / n * 100

print(f"\n--- Customer Satisfaction ---")
print(f"Reviews rated {SATISFACTION_THRESHOLD} stars or higher: {satisfied} / {n}")
print(f"Customer satisfaction level: {satisfaction_pct:.2f}%")

# ---------------------------------------------------------------
# 4. Confidence interval for the true population mean rating
#    (t-distribution, population std unknown)
# ---------------------------------------------------------------
standard_error = std_rating / np.sqrt(n)
t_critical = stats.t.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2, df=n - 1)
margin_of_error = t_critical * standard_error

ci_lower = mean_rating - margin_of_error
ci_upper = mean_rating + margin_of_error

print(f"\n--- {CONFIDENCE_LEVEL * 100:.0f}% Confidence Interval for Mean Rating ---")
print(f"Standard error:  {standard_error:.4f}")
print(f"t-critical:      {t_critical:.4f}")
print(f"Margin of error: {margin_of_error:.4f}")
print(f"{CONFIDENCE_LEVEL * 100:.0f}% CI for the true population mean rating: "
      f"({ci_lower:.3f}, {ci_upper:.3f})")

# Cross-check using SciPy's built-in interval function
ci_scipy = stats.t.interval(CONFIDENCE_LEVEL, df=n - 1,
                             loc=mean_rating, scale=standard_error)
print(f"(SciPy stats.t.interval cross-check: "
      f"({ci_scipy[0]:.3f}, {ci_scipy[1]:.3f}))")
