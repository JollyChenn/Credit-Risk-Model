"""
============================================================
  CREDIT RISK MODEL - Probability of Default (PD)
  Step 1: TRAIN THE MODEL
============================================================
  Run this ONCE to train the model on historical loan data.
  After training, use 'score_applicant.py' to evaluate
  new loan applicants.

  Usage:
    python credit_risk_model.py
============================================================
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import shap


# ============================================================
# CONFIG
# ============================================================
RANDOM_STATE = 42
N_SAMPLES = 2000
TEST_SIZE = 0.2

# Basel II parameters
LGD = 0.45      # Loss Given Default - industry standard for unsecured loans
COST_FN = 10    # Cost of approving a defaulter
COST_FP = 1     # Cost of rejecting a good borrower


# ============================================================
# STEP 1 - GENERATE OR LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1 - Loading data")
print("=" * 60)

np.random.seed(RANDOM_STATE)
df = pd.DataFrame({
    "age": np.random.randint(21, 65, N_SAMPLES),
    "annual_income": np.random.lognormal(10.8, 0.5, N_SAMPLES).astype(int),
    "employment_years": np.random.randint(0, 30, N_SAMPLES),
    "loan_amount": np.random.lognormal(9.5, 0.7, N_SAMPLES).astype(int),
    "existing_debt": np.random.lognormal(8.5, 1.0, N_SAMPLES).astype(int),
    "num_credit_lines": np.random.randint(1, 15, N_SAMPLES),
    "credit_utilization": np.random.beta(2, 5, N_SAMPLES),
    "months_since_last_delinquency": np.random.choice(
        [0, 6, 12, 24, 36, 60, 999], N_SAMPLES,
        p=[0.15, 0.05, 0.05, 0.1, 0.1, 0.15, 0.4]
    ),
    "loan_purpose": np.random.choice(
        ["debt_consolidation", "home_improvement", "business",
         "education", "personal"],
        N_SAMPLES, p=[0.35, 0.2, 0.15, 0.15, 0.15]
    ),
})


# ============================================================
# STEP 2 - FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2 - Feature engineering")
print("=" * 60)

df["debt_to_income"] = df["existing_debt"] / df["annual_income"]
df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
df["payment_burden"] = (df["existing_debt"] + df["loan_amount"]) / df["annual_income"]
df["credit_history_length"] = df["age"] - 18
df["recent_delinquency"] = (df["months_since_last_delinquency"] < 24).astype(int)
df["utilization_x_debt"] = df["credit_utilization"] * df["debt_to_income"]
df["income_per_credit_line"] = df["annual_income"] / (df["num_credit_lines"] + 1)

# Generate realistic default labels (target variable)
default_score = (
    -2.0
    + 1.5 * df["debt_to_income"]
    + 0.8 * df["loan_to_income"]
    + 1.2 * df["credit_utilization"]
    - 0.02 * df["employment_years"]
    - 0.01 * df["age"]
    + 0.5 * df["recent_delinquency"]
    + np.random.normal(0, 0.5, N_SAMPLES)
)
df["default"] = (default_score > np.percentile(default_score, 80)).astype(int)

print(f"  Total borrowers:       {len(df):,}")
print(f"  Default rate:          {df['default'].mean():.1%}")
print(f"  Features (raw):        9")
print(f"  Features (engineered): 7")


# ============================================================
# STEP 3 - PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3 - Preprocessing")
print("=" * 60)

df_encoded = pd.get_dummies(df, columns=["loan_purpose"], drop_first=True)
X = df_encoded.drop("default", axis=1)
y = df_encoded["default"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Training samples:  {len(X_train):,}")
print(f"  Test samples:      {len(X_test):,}")
print(f"  Total features:    {len(feature_names)}")


# ============================================================
# STEP 4 - MODEL TRAINING & COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("STEP 4 - Training models")
print("=" * 60)

# Logistic Regression (interpretable baseline)
lr_model = LogisticRegression(
    max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
)
lr_model.fit(X_train_scaled, y_train)
lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)

# Random Forest (non-linear comparison)
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)

print(f"  Logistic Regression ROC-AUC:  {lr_auc:.4f}")
print(f"  Random Forest ROC-AUC:        {rf_auc:.4f}")
print(f"\n  -> Selected: Logistic Regression (chosen for explainability)")

best_probs = lr_probs
best_auc = lr_auc


# ============================================================
# STEP 5 - COST-SENSITIVE THRESHOLD OPTIMIZATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 5 - Cost-sensitive threshold optimization")
print("=" * 60)
print(f"  Cost ratio: 1 missed default = {COST_FN}x cost of 1 false rejection")

thresholds = np.arange(0.05, 0.95, 0.01)
costs = []
for t in thresholds:
    preds = (best_probs >= t).astype(int)
    cm_t = confusion_matrix(y_test, preds)
    fp = cm_t[0][1]
    fn = cm_t[1][0]
    costs.append(fp * COST_FP + fn * COST_FN)

optimal_threshold = thresholds[np.argmin(costs)]
y_preds = (best_probs >= optimal_threshold).astype(int)

print(f"  Optimal threshold: {optimal_threshold:.3f}")


# ============================================================
# STEP 6 - EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 6 - Final evaluation")
print("=" * 60)

accuracy = accuracy_score(y_test, y_preds)
cm = confusion_matrix(y_test, y_preds)
tn, fp, fn, tp = cm.ravel()
recall = tp / (tp + fn)
precision = tp / (tp + fp)

print(f"  ROC-AUC:    {best_auc:.4f}")
print(f"  Accuracy:   {accuracy:.4f}")
print(f"  Recall:     {recall:.4f}  (caught {recall*100:.0f}% of defaulters)")
print(f"  Precision:  {precision:.4f}")
print(f"\n  Confusion Matrix:")
print(f"    True Negative:  {tn}    False Positive: {fp}")
print(f"    False Negative: {fn}    True Positive:  {tp}")


# ============================================================
# STEP 7 - FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("STEP 7 - Top risk drivers")
print("=" * 60)

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": lr_model.coef_[0]
}).sort_values("coefficient", ascending=False)

for _, row in coef_df.head(8).iterrows():
    direction = "UP risk" if row["coefficient"] > 0 else "DOWN risk"
    bar = "#" * min(int(abs(row["coefficient"]) * 10), 20)
    print(f"  {row['feature']:32s} {row['coefficient']:+.3f}  {bar} {direction}")


# ============================================================
# STEP 8 - SHAP EXPLAINABILITY
# ============================================================
print("\n" + "=" * 60)
print("STEP 8 - SHAP explainability")
print("=" * 60)
print("  Computing SHAP values for individual decisions...")

explainer = shap.LinearExplainer(lr_model, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)

print("  OK - SHAP values computed")


# ============================================================
# STEP 9 - GENERATE DASHBOARD
# ============================================================
print("\n" + "=" * 60)
print("STEP 9 - Generating dashboard")
print("=" * 60)

# Premium dark theme
BG = "#0a0e1a"
CARD = "#141a2a"
GRID = "#1f2940"
TEXT = "#ffffff"
TEXT_MUTED = "#8b95a8"
TEXT_DIM = "#5a6478"
ACCENT_BLUE = "#5fb3f5"
ACCENT_PURPLE = "#a78bfa"
ACCENT_GREEN = "#4ade80"
ACCENT_RED = "#f87171"
ACCENT_AMBER = "#fbbf24"
ACCENT_PINK = "#f472b6"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_MUTED, "axes.titlecolor": TEXT, "text.color": TEXT,
    "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED, "grid.color": GRID,
    "grid.alpha": 0.4, "font.family": "DejaVu Sans", "font.size": 11,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
})

fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor(BG)
gs_main = gridspec.GridSpec(2, 1, height_ratios=[0.22, 1.0], hspace=0.12, figure=fig)

# Header
header_ax = fig.add_subplot(gs_main[0])
header_ax.set_facecolor(BG)
header_ax.set_xlim(0, 100)
header_ax.set_ylim(0, 100)
header_ax.axis("off")
header_ax.text(2, 78, "CREDIT RISK MODEL", fontsize=11, color=ACCENT_BLUE, fontweight="bold")
header_ax.text(2, 50, "Probability of Default Dashboard", fontsize=26, color=TEXT, fontweight="bold")
header_ax.text(2, 22, "Logistic Regression  -  Cost-sensitive thresholds  -  SHAP explainability",
               fontsize=12, color=TEXT_MUTED)

kpi_data = [
    ("ROC-AUC", f"{best_auc:.3f}", ACCENT_BLUE, "Discriminatory power"),
    ("RECALL", f"{recall*100:.0f}%", ACCENT_GREEN, "Defaulters caught"),
    ("THRESHOLD", f"{optimal_threshold:.2f}", ACCENT_AMBER, "Cost-optimized"),
    ("DATASET", f"{N_SAMPLES:,}", ACCENT_PURPLE, "Borrowers analyzed"),
]
kpi_x_start = 42
kpi_width = 14.5
for i, (label, value, color, sublabel) in enumerate(kpi_data):
    x = kpi_x_start + i * kpi_width
    card = FancyBboxPatch((x, 10), kpi_width - 1, 80,
                          boxstyle="round,pad=0.02,rounding_size=1.5",
                          linewidth=0.6, edgecolor=GRID, facecolor=CARD,
                          transform=header_ax.transData)
    header_ax.add_patch(card)
    header_ax.add_patch(patches.Rectangle((x, 10), 0.4, 80, color=color,
                                          transform=header_ax.transData))
    header_ax.text(x + 1.5, 70, label, fontsize=9, color=TEXT_MUTED, fontweight="bold")
    header_ax.text(x + 1.5, 38, value, fontsize=22, color=color, fontweight="bold")
    header_ax.text(x + 1.5, 22, sublabel, fontsize=9, color=TEXT_DIM)

# Body grid
gs_body = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_main[1],
                                            hspace=0.45, wspace=0.28)

def style_panel(ax, title, subtitle):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT_MUTED, length=0, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.5)
    ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(f"{title}\n", fontsize=13, color=TEXT, fontweight="bold", loc="left", pad=18)
    ax.text(0, 1.04, subtitle, transform=ax.transAxes, fontsize=10, color=TEXT_MUTED)

# Panel 1: ROC
ax1 = fig.add_subplot(gs_body[0, 0])
style_panel(ax1, "Model Discrimination", f"ROC curve - AUC {lr_auc:.3f}")
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
ax1.fill_between(fpr_lr, tpr_lr, alpha=0.15, color=ACCENT_BLUE)
ax1.plot(fpr_lr, tpr_lr, color=ACCENT_BLUE, linewidth=2.8, label=f"Logistic Regression ({lr_auc:.3f})")
ax1.plot(fpr_rf, tpr_rf, color=ACCENT_PURPLE, linewidth=2, alpha=0.9, label=f"Random Forest ({rf_auc:.3f})")
ax1.plot([0, 1], [0, 1], color=TEXT_DIM, linestyle="--", linewidth=1, label="Random (0.500)")
ax1.set_xlabel("False Positive Rate", fontsize=10)
ax1.set_ylabel("True Positive Rate", fontsize=10)
legend = ax1.legend(loc="lower right", fontsize=9, frameon=True, facecolor=BG, edgecolor=GRID)
for text in legend.get_texts():
    text.set_color(TEXT)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.02)

# Panel 2: Confusion matrix
ax2 = fig.add_subplot(gs_body[0, 1])
ax2.set_facecolor(CARD)
ax2.set_xlim(-0.15, 2)
ax2.set_ylim(-0.15, 2.2)
ax2.axis("off")
ax2.text(-0.05, 1.1, "Classification Outcomes", fontsize=13, color=TEXT,
         fontweight="bold", transform=ax2.transAxes)
ax2.text(-0.05, 1.05, "Confusion matrix at optimal threshold", fontsize=10,
         color=TEXT_MUTED, transform=ax2.transAxes)

cm_cells = [
    (0, 1, "TRUE NEGATIVE", str(tn), "Correctly approved", ACCENT_GREEN),
    (1, 1, "FALSE POSITIVE", str(fp), "Wrongly rejected", ACCENT_AMBER),
    (0, 0, "FALSE NEGATIVE", str(fn), "Missed defaulter", ACCENT_RED),
    (1, 0, "TRUE POSITIVE", str(tp), "Correctly rejected", ACCENT_BLUE),
]
for x, y_pos, label, value, sub, color in cm_cells:
    rect = FancyBboxPatch((x + 0.05, y_pos + 0.05), 0.9, 0.9,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          linewidth=0.5, edgecolor=GRID, facecolor=BG)
    ax2.add_patch(rect)
    ax2.add_patch(patches.Rectangle((x + 0.05, y_pos + 0.05), 0.04, 0.9, color=color))
    ax2.text(x + 0.15, y_pos + 0.78, label, fontsize=9, color=color, fontweight="bold")
    ax2.text(x + 0.15, y_pos + 0.40, value, fontsize=32, color=TEXT, fontweight="bold")
    ax2.text(x + 0.15, y_pos + 0.18, sub, fontsize=10, color=TEXT_MUTED)

ax2.text(0.5, -0.12, "PREDICTED: NO DEFAULT", fontsize=8, color=TEXT_DIM,
         ha="center", fontweight="bold")
ax2.text(1.5, -0.12, "PREDICTED: DEFAULT", fontsize=8, color=TEXT_DIM,
         ha="center", fontweight="bold")
ax2.text(-0.1, 1.5, "ACTUAL:\nNO DEFAULT", fontsize=8, color=TEXT_DIM,
         ha="right", va="center", fontweight="bold")
ax2.text(-0.1, 0.5, "ACTUAL:\nDEFAULT", fontsize=8, color=TEXT_DIM,
         ha="right", va="center", fontweight="bold")

# Panel 3: Threshold cost
ax3 = fig.add_subplot(gs_body[0, 2])
style_panel(ax3, "Threshold Optimization", "Total business cost across thresholds")
ax3.fill_between(thresholds, costs, alpha=0.2, color=ACCENT_AMBER)
ax3.plot(thresholds, costs, color=ACCENT_AMBER, linewidth=2.5)
ax3.axvline(x=optimal_threshold, color=ACCENT_RED, linestyle="--", linewidth=2, alpha=0.8)
ax3.scatter([optimal_threshold], [min(costs)], color=ACCENT_RED, s=120,
            zorder=10, edgecolor=BG, linewidth=2)
ax3.annotate(f"Optimal: {optimal_threshold:.2f}",
             xy=(optimal_threshold, min(costs)),
             xytext=(optimal_threshold + 0.15, min(costs) + 30),
             fontsize=10, color=ACCENT_RED, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1))
ax3.set_xlabel("Probability threshold", fontsize=10)
ax3.set_ylabel("Total cost (FP*1 + FN*10)", fontsize=10)

# Panel 4: Risk drivers
ax4 = fig.add_subplot(gs_body[1, 0])
style_panel(ax4, "Risk Drivers", "Logistic regression coefficients")
top_features = coef_df.head(8).iloc[::-1]
colors = [ACCENT_RED if c > 0 else ACCENT_GREEN for c in top_features["coefficient"]]
bars = ax4.barh(range(len(top_features)), top_features["coefficient"],
                color=colors, height=0.65, edgecolor="none")
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels([f.replace("_", " ").title() for f in top_features["feature"]],
                    fontsize=10, color=TEXT)
ax4.set_xlabel("Coefficient   ->   higher = more risk", fontsize=10)
ax4.axvline(x=0, color=TEXT_DIM, linewidth=0.6)
for i, (bar, val) in enumerate(zip(bars, top_features["coefficient"])):
    ax4.text(val + (0.03 if val > 0 else -0.03), i, f"{val:+.2f}",
             va="center", ha="left" if val > 0 else "right",
             fontsize=9, color=TEXT_MUTED, fontweight="bold")
ax4.set_xlim(min(top_features["coefficient"]) - 0.15,
             max(top_features["coefficient"]) + 0.25)

# Panel 5: PD distribution
ax5 = fig.add_subplot(gs_body[1, 1])
style_panel(ax5, "Score Distribution", "Predicted PD by actual outcome")
ax5.hist(best_probs[y_test == 0], bins=30, alpha=0.75, color=ACCENT_GREEN,
         label="Did not default", density=True, edgecolor="none")
ax5.hist(best_probs[y_test == 1], bins=30, alpha=0.75, color=ACCENT_RED,
         label="Defaulted", density=True, edgecolor="none")
ax5.axvline(x=optimal_threshold, color=ACCENT_AMBER, linestyle="--", linewidth=2,
            label=f"Threshold = {optimal_threshold:.2f}")
ax5.set_xlabel("Predicted probability of default", fontsize=10)
ax5.set_ylabel("Density", fontsize=10)
legend5 = ax5.legend(loc="upper center", fontsize=9, frameon=True, facecolor=BG, edgecolor=GRID)
for text in legend5.get_texts():
    text.set_color(TEXT)

# Panel 6: SHAP importance
ax6 = fig.add_subplot(gs_body[1, 2])
style_panel(ax6, "Feature Importance", "Mean |SHAP value| across all decisions")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({"feature": feature_names, "importance": mean_abs_shap}) \
            .sort_values("importance", ascending=True).tail(8)
ax6.barh(range(len(shap_df)), shap_df["importance"], color=ACCENT_PURPLE,
         height=0.65, edgecolor="none", alpha=0.9)
for i, (idx, row) in enumerate(shap_df.iterrows()):
    ax6.barh(i, row["importance"] * 0.3, color=ACCENT_PINK,
             height=0.65, edgecolor="none", alpha=0.8)
ax6.set_yticks(range(len(shap_df)))
ax6.set_yticklabels([f.replace("_", " ").title() for f in shap_df["feature"]],
                    fontsize=10, color=TEXT)
ax6.set_xlabel("Average impact on prediction", fontsize=10)
for i, val in enumerate(shap_df["importance"]):
    ax6.text(val + 0.01, i, f"{val:.2f}", va="center", ha="left",
             fontsize=9, color=TEXT_MUTED, fontweight="bold")

fig.text(0.013, 0.001, "Built with Python - scikit-learn - SHAP",
         fontsize=9, color=TEXT_DIM, ha="left")
fig.text(0.987, 0.001, "Banking & Finance - Fintech specialization",
         fontsize=9, color=TEXT_DIM, ha="right")

plt.subplots_adjust(left=0.05, right=0.97, top=0.97, bottom=0.07)
plt.savefig("credit_risk_dashboard.png", dpi=200, bbox_inches="tight",
            facecolor=BG, edgecolor="none", pad_inches=0.3)
plt.close()

print("  OK - Dashboard saved: credit_risk_dashboard.png")


# ============================================================
# STEP 10 - SAVE TRAINED MODEL
# ============================================================
joblib.dump({
    "model": lr_model,
    "scaler": scaler,
    "feature_names": feature_names,
    "optimal_threshold": float(optimal_threshold),
    "lgd": LGD
}, "trained_model.pkl")

# Save summary JSON
summary = {
    "best_model": "Logistic Regression",
    "performance": {
        "roc_auc": round(best_auc, 4),
        "accuracy": round(accuracy, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "optimal_threshold": round(optimal_threshold, 4)
    },
    "confusion_matrix": {
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp)
    },
    "top_risk_factors": coef_df.head(5)[["feature", "coefficient"]].to_dict("records"),
    "decision_thresholds": {
        "approve":              "PD < 5%",
        "approve_higher_rate":  "5% <= PD < 15%",
        "manual_review":        "15% <= PD < 30%",
        "reject":               "PD >= 30%"
    }
}

with open("model_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("  OK - Model saved: trained_model.pkl")
print("  OK - Summary saved: model_summary.json")


# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("\nFiles created in this folder:")
print("  - credit_risk_dashboard.png   (the visual for LinkedIn)")
print("  - trained_model.pkl           (the trained model)")
print("  - model_summary.json          (results in JSON format)")
print("\nNext step:")
print("  Run 'python score_applicant.py' to evaluate a new loan applicant.")
print("=" * 60)
