"""
Q21. Point Estimation and Confidence Interval for a Population Mean
-----------------------------------------------------------------------
Scenario: A scientist is researching rare elements found in a specific
region. The goal is to estimate the average concentration of a rare
element using a random sample of measurements, drawn from
"rare_elements.csv" (one concentration measurement per row).

This program:
    1. Loads the concentration data using NumPy.
    2. Lets the user input the sample size, confidence level, and the
       desired level of precision (margin of error).
    3. Draws a random sample of the requested size from the data.
    4. Calculates the point estimate (sample mean) of the population
       mean concentration.
    5. Calculates the confidence interval for the population mean.
    6. Uses the desired precision to work out the minimum sample size
       that would be needed to achieve that precision at the chosen
       confidence level.
"""

import os
import numpy as np
from scipy import stats

CSV_FILE = "rare_elements.csv"

# ---------------------------------------------------------------
# Create a sample rare_elements.csv if it does not already exist
# (so the script can be run/tested end-to-end).
# Remove this block once you have your real data file.
# ---------------------------------------------------------------
if not os.path.exists(CSV_FILE):
    rng = np.random.default_rng(7)
    concentrations = rng.normal(loc=5.20, scale=1.15, size=300)
    concentrations = np.round(np.abs(concentrations), 3)
    np.savetxt(CSV_FILE, concentrations, delimiter=",", header="concentration",
               comments="")
    print(f"'{CSV_FILE}' not found — a sample file was generated for demo purposes.\n")

# ---------------------------------------------------------------
# 1. Load the data with NumPy
# ---------------------------------------------------------------
data = np.genfromtxt(CSV_FILE, delimiter=",", skip_header=1)
population_size = data.size
print(f"Loaded {population_size} concentration measurements from '{CSV_FILE}'.\n")

# ---------------------------------------------------------------
# 2. Get user input: sample size, confidence level, desired precision
# ---------------------------------------------------------------
sample_size = int(input("Enter the sample size: "))
confidence_level = float(input("Enter the confidence level (e.g. 0.95 for 95%): "))
desired_precision = float(input("Enter the desired level of precision "
                                 "(margin of error, e.g. 0.1): "))

if sample_size > population_size:
    raise ValueError("Sample size cannot exceed the number of available measurements.")

# ---------------------------------------------------------------
# 3. Draw a random sample of the requested size
# ---------------------------------------------------------------
rng = np.random.default_rng()
sample = rng.choice(data, size=sample_size, replace=False)

# ---------------------------------------------------------------
# 4. Point estimation of the population mean
# ---------------------------------------------------------------
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)   # sample standard deviation

print("\n--- Point Estimation ---")
print(f"Sample size (n):        {sample_size}")
print(f"Sample mean (point estimate of population mean): {sample_mean:.4f}")
print(f"Sample standard deviation:                        {sample_std:.4f}")

# ---------------------------------------------------------------
# 5. Confidence interval for the population mean (t-distribution,
#    since the population standard deviation is unknown)
# ---------------------------------------------------------------
alpha = 1 - confidence_level
t_critical = stats.t.ppf(1 - alpha / 2, df=sample_size - 1)
standard_error = sample_std / np.sqrt(sample_size)
margin_of_error = t_critical * standard_error

ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

print("\n--- Confidence Interval ---")
print(f"Confidence level:        {confidence_level * 100:.1f}%")
print(f"t-critical value:        {t_critical:.4f}")
print(f"Margin of error:         {margin_of_error:.4f}")
print(f"{confidence_level * 100:.1f}% Confidence Interval for the population mean: "
      f"({ci_lower:.4f}, {ci_upper:.4f})")

# ---------------------------------------------------------------
# 6. Minimum sample size needed to achieve the desired precision
#    n = (z * sigma / E)^2   -- using the sample std as an estimate
#    of the population std, and the z critical value for the
#    requested confidence level.
# ---------------------------------------------------------------
z_critical = stats.norm.ppf(1 - alpha / 2)
required_n = (z_critical * sample_std / desired_precision) ** 2
required_n = int(np.ceil(required_n))

print("\n--- Required Sample Size for Desired Precision ---")
print(f"Desired margin of error: {desired_precision}")
print(f"z-critical value:        {z_critical:.4f}")
print(f"Minimum sample size required to achieve this precision "
      f"at {confidence_level * 100:.1f}% confidence: {required_n}")
