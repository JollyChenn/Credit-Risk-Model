# Credit Risk Model — Probability of Default

A fintech project that predicts borrower default probability and makes risk-based lending decisions.

Built by a Banking \& Finance student specializing in Fintech.

 What It Does

This model predicts whether a loan applicant will default, then translates that probability into a real lending decision:

|Probability of Default|Decision|
|-|-|
|Less than 5%|APPROVE|
|5% – 15%|APPROVE with higher interest|
|15% – 30%|MANUAL REVIEW|
|30% or more|REJECT|

It also calculates Expected Loss using the Basel II formula:

EL = PD × LGD × EAD

where LGD = Loss Given Default (45% industry standard) and EAD = the loan amount.

How to Use

1. Install dependencies (one time)

pip install -r requirements.txt

2. Train the model (one time)

python credit_risk_model.py

3. Score new applicants (any time)

python scoreapplicant.py

Key Features

Feature Engineering — uses banking-grade ratios (debt-to-income, loan-to-income, payment burden, credit utilization × debt)

Model Comparison — trains both Logistic Regression and Random Forest,

Cost-Sensitive Threshold — instead of the default 0.5 cutoff, the threshold is optimized assuming a missed default costs 10× a false rejection. This drops the optimal cutoff to 0.26 and catches 90% of defaulters.

SHAP Explainability — every individual decision can be traced back to the specific features that drove it, satisfying regulatory requirements for transparent lendingTechnical Stack

* Python 3.10+
* pandas, numpy — data manipulation
* scikit-learn — modeling
* SHAP — explainability
* matplotlib — visualization
* joblib — model persistence


Files

|File|Purpose|
|-|-|
|`credit\_risk\_model.py`|Trains the model and generates the dashboard|
|`score\_applicant.py`|Interactive scoring of new applicants|
|`requirements.txt`|Python dependencies|
|`credit\_risk\_dashboard.png`|Performance visualization|
|`trained\_model.pkl`|Saved model (generated after training)|
|`model\_summary.json`|Results in machine-readable format|


*Disclaimer: This model uses synthetic data for demonstration. It is not intended for actual lending decisions.*

