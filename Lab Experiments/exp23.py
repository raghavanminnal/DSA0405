"""
Q23. Clinical Trial Hypothesis Testing: Placebo vs New Drug
-----------------------------------------------------------------------
Scenario: A researcher in a medical lab is investigating the
effectiveness of a new treatment. Data was collected from a clinical
trial with two groups: a control group receiving a placebo, and a
treatment group receiving the new drug. The goal is to analyze the
data using hypothesis testing, calculate the p-value to determine if
the new treatment has a statistically significant effect compared to
the placebo, and visualize the data and the p-value using Matplotlib.

Hypotheses:
    H0 (null):        There is no difference in the mean outcome
                       between the placebo group and the treatment group.
    H1 (alternative):  The treatment group's mean outcome is
                       significantly different from the placebo group's.

This program:
    1. Loads (or simulates) the control and treatment group data.
    2. Performs an independent two-sample t-test.
    3. Reports the t-statistic and p-value, and states the conclusion
       at a 5% significance level.
    4. Visualizes the two groups (boxplot comparison) and the p-value
       relative to the significance threshold.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

CSV_FILE = "clinical_trial.csv"
ALPHA = 0.05   # significance level

# ---------------------------------------------------------------
# Create a sample clinical_trial.csv if it does not already exist
# (so the script can be run/tested end-to-end).
# Remove this block once you have your real data file.
# Expected columns: 'Group' ('Control' / 'Treatment'), 'Outcome'
# (e.g. an improvement score / recovery measure).
# ---------------------------------------------------------------
if not os.path.exists(CSV_FILE):
    rng = np.random.default_rng(3)
    n_per_group = 40
    control = rng.normal(loc=50, scale=8, size=n_per_group)
    treatment = rng.normal(loc=56, scale=8, size=n_per_group)
    sample = pd.DataFrame({
        "Group": ["Control"] * n_per_group + ["Treatment"] * n_per_group,
        "Outcome": np.round(np.concatenate([control, treatment]), 2)
    })
    sample.to_csv(CSV_FILE, index=False)
    print(f"'{CSV_FILE}' not found — a sample file was generated for demo purposes.\n")

# ---------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------
df = pd.read_csv(CSV_FILE)
control_group = df.loc[df["Group"] == "Control", "Outcome"]
treatment_group = df.loc[df["Group"] == "Treatment", "Outcome"]

print(f"Control group:   n={len(control_group)}, "
      f"mean={control_group.mean():.3f}, std={control_group.std():.3f}")
print(f"Treatment group: n={len(treatment_group)}, "
      f"mean={treatment_group.mean():.3f}, std={treatment_group.std():.3f}\n")

# ---------------------------------------------------------------
# 2. Independent two-sample t-test (Welch's t-test: unequal variances)
# ---------------------------------------------------------------
t_stat, p_value = stats.ttest_ind(treatment_group, control_group, equal_var=False)

print("--- Hypothesis Test: Independent Two-Sample t-test ---")
print(f"H0: No difference between control and treatment group means")
print(f"H1: Treatment group mean differs from control group mean")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.6f}")

# ---------------------------------------------------------------
# 3. Conclusion at the chosen significance level
# ---------------------------------------------------------------
if p_value < ALPHA:
    conclusion = (f"p-value ({p_value:.6f}) < alpha ({ALPHA}) -> Reject H0.\n"
                  f"The new treatment has a statistically significant effect "
                  f"compared to the placebo.")
else:
    conclusion = (f"p-value ({p_value:.6f}) >= alpha ({ALPHA}) -> Fail to reject H0.\n"
                  f"There is not enough evidence that the new treatment has a "
                  f"statistically significant effect compared to the placebo.")
print(f"\nConclusion (alpha = {ALPHA}):\n{conclusion}\n")

# ---------------------------------------------------------------
# 4. Visualization: boxplot comparison + p-value bar
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# -- Boxplot comparing the two groups --
axes[0].boxplot([control_group, treatment_group],
                 tick_labels=["Control (Placebo)", "Treatment (New Drug)"],
                 patch_artist=True,
                 boxprops=dict(facecolor="lightblue"))
axes[0].set_title("Outcome Distribution: Control vs Treatment")
axes[0].set_ylabel("Outcome")

# -- p-value visualization against the significance threshold --
axes[1].bar(["p-value"], [p_value], color="darkorange", width=0.4)
axes[1].axhline(y=ALPHA, color="red", linestyle="--",
                 label=f"Significance level (alpha = {ALPHA})")
axes[1].set_ylim(0, max(p_value, ALPHA) * 1.5 + 0.01)
axes[1].set_title("p-value vs Significance Threshold")
axes[1].set_ylabel("p-value")
axes[1].text(0, p_value, f"{p_value:.4f}", ha="center", va="bottom")
axes[1].legend()

plt.tight_layout()
plt.savefig("clinical_trial_ttest.png", dpi=150)
plt.close()

print("Saved: clinical_trial_ttest.png")
