"""
============================================================
  SCORE NEW APPLICANT - Interactive Mode
============================================================
  This script evaluates a NEW loan applicant.
  Just answer the questions when prompted.

  Usage:
    python score_applicant.py
============================================================
"""

import joblib
import pandas as pd
import numpy as np
import os
import sys


# ============================================================
# Helper: ask the user a question with input validation
# ============================================================
def ask_number(prompt, min_val=None, max_val=None, decimals=False):
    """Ask the user for a number, validate it, return the value."""
    while True:
        try:
            answer = input(f"  {prompt}: ").strip()
            value = float(answer) if decimals else int(answer)
            if min_val is not None and value < min_val:
                print(f"    [!] Must be at least {min_val}. Try again.")
                continue
            if max_val is not None and value > max_val:
                print(f"    [!] Must be at most {max_val}. Try again.")
                continue
            return value
        except ValueError:
            print("    [!] Please enter a valid number.")


def ask_choice(prompt, choices):
    """Ask the user to pick from a list of choices."""
    print(f"  {prompt}:")
    for i, choice in enumerate(choices, 1):
        print(f"    {i}. {choice.replace('_', ' ').title()}")
    while True:
        try:
            answer = input(f"  Enter number (1-{len(choices)}): ").strip()
            idx = int(answer) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
            print(f"    [!] Please enter a number between 1 and {len(choices)}.")
        except ValueError:
            print("    [!] Please enter a valid number.")


# ============================================================
# STEP 1 - Load the trained model
# ============================================================
print("\n" + "=" * 60)
print("  CREDIT RISK MODEL - LOAN APPLICATION SCORING")
print("=" * 60)

if not os.path.exists("trained_model.pkl"):
    print("\n[!] ERROR: 'trained_model.pkl' not found.")
    print("    Please run 'python credit_risk_model.py' first to train the model.\n")
    sys.exit(1)

print("\nLoading trained model...")
bundle = joblib.load("trained_model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]
optimal_threshold = bundle["optimal_threshold"]
LGD = bundle["lgd"]
print("Model ready.\n")


# ============================================================
# STEP 2 - Collect applicant info from user
# ============================================================
print("=" * 60)
print("  ENTER APPLICANT INFORMATION")
print("=" * 60)
print()

age = ask_number("Age (years)", min_val=18, max_val=100)
annual_income = ask_number("Annual income (in your local currency)", min_val=1)
employment_years = ask_number("Years of employment", min_val=0, max_val=60)
loan_amount = ask_number("Loan amount requested", min_val=1)
existing_debt = ask_number("Existing debt (current outstanding)", min_val=0)
num_credit_lines = ask_number("Number of active credit lines", min_val=0, max_val=50)

print()
credit_utilization_pct = ask_number(
    "Credit utilization (% of available credit currently used, 0-100)",
    min_val=0, max_val=100, decimals=True
)
credit_utilization = credit_utilization_pct / 100  # convert to 0-1 range

print()
print("  How long since the last late payment?")
print("    Enter 0 if currently late")
print("    Enter 999 if no late payments ever")
months_since_last_delinquency = ask_number(
    "Months since last delinquency",
    min_val=0, max_val=999
)

print()
loan_purpose = ask_choice("Loan purpose", [
    "debt_consolidation", "home_improvement", "business",
    "education", "personal"
])


# ============================================================
# STEP 3 - Build the feature row
# ============================================================
applicant = {
    "age": age,
    "annual_income": annual_income,
    "employment_years": employment_years,
    "loan_amount": loan_amount,
    "existing_debt": existing_debt,
    "num_credit_lines": num_credit_lines,
    "credit_utilization": credit_utilization,
    "months_since_last_delinquency": months_since_last_delinquency,
    "loan_purpose": loan_purpose
}

df = pd.DataFrame([applicant])
df["debt_to_income"] = df["existing_debt"] / df["annual_income"]
df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
df["payment_burden"] = (df["existing_debt"] + df["loan_amount"]) / df["annual_income"]
df["credit_history_length"] = df["age"] - 18
df["recent_delinquency"] = (df["months_since_last_delinquency"] < 24).astype(int)
df["utilization_x_debt"] = df["credit_utilization"] * df["debt_to_income"]
df["income_per_credit_line"] = df["annual_income"] / (df["num_credit_lines"] + 1)

# One-hot encode loan_purpose
df = pd.get_dummies(df, columns=["loan_purpose"], drop_first=True)

# Add any missing columns (purposes not selected) as 0
for col in feature_names:
    if col not in df.columns:
        df[col] = 0

df = df[feature_names]


# ============================================================
# STEP 4 - Score
# ============================================================
X_scaled = scaler.transform(df)
pd_score = model.predict_proba(X_scaled)[0, 1]
expected_loss = pd_score * LGD * loan_amount


# ============================================================
# STEP 5 - Decision
# ============================================================
if pd_score < 0.05:
    decision = "APPROVE"
    risk_tier = "Low Risk"
    note = "Standard interest rate."
elif pd_score < 0.15:
    decision = "APPROVE (Higher Interest Rate)"
    risk_tier = "Medium Risk"
    note = "Approve with risk-adjusted pricing."
elif pd_score < 0.30:
    decision = "MANUAL REVIEW"
    risk_tier = "High Risk"
    note = "Send to human underwriter for additional review."
else:
    decision = "REJECT"
    risk_tier = "Very High Risk"
    note = "Default probability too high to approve."


# ============================================================
# STEP 6 - Show key risk factors (top 3)
# ============================================================
contributions = []
for i, fname in enumerate(feature_names):
    coef = model.coef_[0][i]
    val_scaled = X_scaled[0][i]
    contribution = coef * val_scaled
    contributions.append((fname, contribution))

contributions.sort(key=lambda x: abs(x[1]), reverse=True)
top_risk_factors = contributions[:3]


# ============================================================
# STEP 7 - Display result
# ============================================================
print("\n" + "=" * 60)
print("  DECISION RESULT")
print("=" * 60)
print(f"\n  Probability of Default:  {pd_score:.2%}")
print(f"  Risk Tier:               {risk_tier}")
print(f"  Decision:                {decision}")
print(f"  Expected Loss:           {expected_loss:,.2f}")
print(f"  ({note})")

print(f"\n  Top 3 factors influencing this decision:")
for fname, contrib in top_risk_factors:
    direction = "increased risk" if contrib > 0 else "decreased risk"
    nice_name = fname.replace("_", " ").title()
    print(f"    - {nice_name:30s}  ({direction})")

print(f"\n  Decision threshold used: {optimal_threshold:.2f}")
print("=" * 60)
print()
