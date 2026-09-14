"""
============================================================
MuleWatch - Step 3: Supervised Fraud Classifier
============================================================
PURPOSE: Train RF, XGBoost, SMOTE-RF to predict 'is_mule'.
INPUT:   ../02_data/processed/account_features.csv
OUTPUT:  ../04_models/*.pkl + ../08_outputs/*.json/png
============================================================
"""

import pandas as pd, numpy as np, json, os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, precision_recall_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED = "../02_data/processed"
MODELS = "../04_models"; os.makedirs(MODELS, exist_ok=True)
OUT = "../08_outputs"; os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(f"{PROCESSED}/account_features.csv")
cols = [c for c in df.columns if c not in ["account_id", "is_mule"]]
X, y = df[cols].values, df["is_mule"].values
print(f"Features: {len(cols)} | Positives: {y.sum()} / {len(y)}")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- Rule baseline ---
rule_pred = ((df["auth_fail_rate"] > 0.3) |
             (df["txn_sub_threshold_ratio"] > 0.5)).astype(int).values[-len(yte):]
print("\n--- Rule baseline ---")
print(classification_report(yte, rule_pred, zero_division=0))

# --- Random Forest ---
print("\n--- Random Forest ---")
rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                            class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(Xtr, ytr)
rf_pr = rf.predict_proba(Xte)[:, 1]
print(classification_report(yte, rf.predict(Xte), zero_division=0))
print(f"ROC-AUC {roc_auc_score(yte, rf_pr):.4f} | PR-AUC {average_precision_score(yte, rf_pr):.4f}")

# --- XGBoost ---
print("\n--- XGBoost ---")
spw = (ytr == 0).sum() / (ytr == 1).sum()
xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                          scale_pos_weight=spw, random_state=42,
                          eval_metric="logloss")
xgb_m.fit(Xtr, ytr)
xgb_pr = xgb_m.predict_proba(Xte)[:, 1]
print(classification_report(yte, xgb_m.predict(Xte), zero_division=0))
print(f"ROC-AUC {roc_auc_score(yte, xgb_pr):.4f} | PR-AUC {average_precision_score(yte, xgb_pr):.4f}")

# --- SMOTE + RF ---
print("\n--- SMOTE + RF ---")
Xtr_s, ytr_s = SMOTE(random_state=42).fit_resample(Xtr, ytr)
rf_s = RandomForestClassifier(n_estimators=200, max_depth=12,
                              random_state=42, n_jobs=-1).fit(Xtr_s, ytr_s)
print(classification_report(yte, rf_s.predict(Xte), zero_division=0))

# --- Feature importance ---
imp = pd.DataFrame({"feature": cols, "importance": rf.feature_importances_}
                   ).sort_values("importance", ascending=False)
print("\nTop 10 features:")
print(imp.head(10).to_string(index=False))
imp.to_csv(f"{OUT}/feature_importance.csv", index=False)

# --- Save ---
joblib.dump(rf, f"{MODELS}/random_forest.pkl")
joblib.dump(xgb_m, f"{MODELS}/xgboost.pkl")
joblib.dump(rf_s, f"{MODELS}/random_forest_smote.pkl")

json.dump({
    "random_forest": {"roc_auc": float(roc_auc_score(yte, rf_pr)),
                      "pr_auc": float(average_precision_score(yte, rf_pr))},
    "xgboost": {"roc_auc": float(roc_auc_score(yte, xgb_pr)),
                "pr_auc": float(average_precision_score(yte, xgb_pr))},
}, open(f"{OUT}/supervised_metrics.json", "w"), indent=2)

plt.figure(figsize=(8, 6))
for name, pr in [("RF", rf_pr), ("XGB", xgb_pr)]:
    p, r, _ = precision_recall_curve(yte, pr)
    plt.plot(r, p, label=f"{name} AP={average_precision_score(yte, pr):.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall - Mule Detection"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/precision_recall_curve.png", dpi=150)
print("\nDone.")