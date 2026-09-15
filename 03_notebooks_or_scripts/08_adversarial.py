"""
============================================================
MuleWatch - Step 8: Adversarial / Robustness Testing
============================================================
PURPOSE:
  Test how the supervised classifier degrades under 3
  evasion strategies: structuring, timing shifts, device
  rotation. Reports recall drop per strategy.
INPUT:
  ../02_data/processed/account_features.csv
  ../04_models/random_forest.pkl
OUTPUT:
  ../08_outputs/adversarial_results.csv
  ../08_outputs/adversarial_metrics.json
============================================================
"""

import pandas as pd
import numpy as np
import os
import joblib
import json

# ---------- PATHS ----------
PROCESSED = "../02_data/processed"
MODELS = "../04_models"
OUT = "../08_outputs"
os.makedirs(OUT, exist_ok=True)

# ---------- LOAD ----------
print("Loading features + trained model...")
df = pd.read_csv(f"{PROCESSED}/account_features.csv")
rf = joblib.load(f"{MODELS}/random_forest.pkl")

cols = [c for c in df.columns if c not in ["account_id", "is_mule"]]
X = df[cols].copy()
y = df["is_mule"].values

# ---------- BASELINE ----------
base_proba = rf.predict_proba(X.values)[:, 1]
base_pred = (base_proba > 0.5).astype(int)
base_recall = ((base_pred == 1) & (y == 1)).sum() / max(y.sum(), 1)
base_precision = ((base_pred == 1) & (y == 1)).sum() / max((base_pred == 1).sum(), 1)

print(f"\nBaseline: recall={base_recall:.4f}, precision={base_precision:.4f}")

results = [{
    "strategy": "baseline",
    "description": "No evasion",
    "recall": round(float(base_recall), 4),
    "precision": round(float(base_precision), 4),
    "recall_drop": 0.0,
}]

# ---------- ADVERSARY 1: Structuring ----------
# Adversary splits transfers further to avoid sub-threshold detection.
print("\n[Adversary 1] Structuring (splitting amounts)")
X_adv = X.copy()
X_adv["txn_sub_threshold_ratio"] = X_adv["txn_sub_threshold_ratio"] * 0.5
adv_pred = (rf.predict_proba(X_adv.values)[:, 1] > 0.5).astype(int)
adv_recall = ((adv_pred == 1) & (y == 1)).sum() / max(y.sum(), 1)
adv_precision = ((adv_pred == 1) & (y == 1)).sum() / max((adv_pred == 1).sum(), 1)
results.append({
    "strategy": "structuring",
    "description": "Split sub-threshold transfers in half",
    "recall": round(float(adv_recall), 4),
    "precision": round(float(adv_precision), 4),
    "recall_drop": round(float(base_recall - adv_recall), 4),
})

# ---------- ADVERSARY 2: Timing shifts ----------
# Adversary slows transfers to defeat the rapid pass-through signal.
print("[Adversary 2] Timing shifts (slower transfers)")
X_adv = X.copy()
X_adv["txn_rapid_ratio"] = X_adv["txn_rapid_ratio"] * 0.3
adv_pred = (rf.predict_proba(X_adv.values)[:, 1] > 0.5).astype(int)
adv_recall = ((adv_pred == 1) & (y == 1)).sum() / max(y.sum(), 1)
adv_precision = ((adv_pred == 1) & (y == 1)).sum() / max((adv_pred == 1).sum(), 1)
results.append({
    "strategy": "timing_shift",
    "description": "Reduce rapid pass-through ratio by 70%",
    "recall": round(float(adv_recall), 4),
    "precision": round(float(adv_precision), 4),
    "recall_drop": round(float(base_recall - adv_recall), 4),
})

# ---------- ADVERSARY 3: Device rotation ----------
# Adversary rotates devices and cleans up auth failures to look normal.
print("[Adversary 3] Device rotation")
X_adv = X.copy()
X_adv["auth_unique_devices"] = X_adv["auth_unique_devices"].clip(0, 2)
X_adv["auth_fail_rate"] = X_adv["auth_fail_rate"] * 0.4
adv_pred = (rf.predict_proba(X_adv.values)[:, 1] > 0.5).astype(int)
adv_recall = ((adv_pred == 1) & (y == 1)).sum() / max(y.sum(), 1)
adv_precision = ((adv_pred == 1) & (y == 1)).sum() / max((adv_pred == 1).sum(), 1)
results.append({
    "strategy": "device_rotation",
    "description": "Rotate devices + reduce auth failure rate",
    "recall": round(float(adv_recall), 4),
    "precision": round(float(adv_precision), 4),
    "recall_drop": round(float(base_recall - adv_recall), 4),
})

# ---------- SAVE ----------
results_df = pd.DataFrame(results)
print("\n--- Adversarial Test Results ---")
print(results_df.to_string(index=False))

results_df.to_csv(f"{OUT}/adversarial_results.csv", index=False)

json.dump({
    "baseline_recall": float(base_recall),
    "baseline_precision": float(base_precision),
    "max_recall_drop": float(results_df["recall_drop"].max()),
}, open(f"{OUT}/adversarial_metrics.json", "w"), indent=2)

print(f"\nSaved: {OUT}/adversarial_results.csv")
print("Adversarial testing complete.")