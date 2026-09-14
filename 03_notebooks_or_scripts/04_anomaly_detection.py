"""
============================================================
MuleWatch - Step 4: Unsupervised Anomaly Detection
============================================================
PURPOSE: Isolation Forest + z-score baselines to flag
         accounts behaving unlike the normal population.
INPUT:   ../02_data/processed/account_features.csv
OUTPUT:  ../04_models/isolation_forest.pkl
         ../02_data/processed/anomaly_scores.csv
         ../08_outputs/anomaly_metrics.json
============================================================
"""

import pandas as pd, numpy as np, joblib, json, os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# ---------- PATHS ----------
PROCESSED = "../02_data/processed"
MODELS = "../04_models"; os.makedirs(MODELS, exist_ok=True)
OUT = "../08_outputs"; os.makedirs(OUT, exist_ok=True)

# ---------- LOAD ----------
df = pd.read_csv(f"{PROCESSED}/account_features.csv")
cols = [c for c in df.columns if c not in ["account_id", "is_mule"]]
X, y = df[cols].values, df["is_mule"].values

# ---------- SCALE ----------
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)

# ---------- ISOLATION FOREST ----------
print("--- Isolation Forest ---")
iso = IsolationForest(n_estimators=200, contamination=0.05,
                      random_state=42, n_jobs=-1).fit(Xs)
iso_pred = (iso.predict(Xs) == -1).astype(int)
iso_scores = -iso.score_samples(Xs)  # higher = more anomalous
print(classification_report(y, iso_pred, zero_division=0))
print(f"ROC-AUC {roc_auc_score(y, iso_scores):.4f}")

# ---------- Z-SCORE BEHAVIOURAL BASELINE ----------
print("\n--- Z-score baseline ---")
keys = ["auth_fail_rate", "txn_sub_threshold_ratio",
        "txn_rapid_ratio", "auth_off_hours_rate", "benef_change_count"]
z = pd.DataFrame()
for c in keys:
    if c in df.columns:
        mu, sd = df[c].mean(), df[c].std()
        z[c] = np.abs((df[c] - mu) / sd) if sd > 0 else 0

df["zscore_anomaly"] = z.mean(axis=1)
thr = df["zscore_anomaly"].quantile(0.95)
df["zscore_flag"] = (df["zscore_anomaly"] > thr).astype(int)
print(f"Threshold = {thr:.3f}")
print(classification_report(y, df["zscore_flag"], zero_division=0))

# ---------- COMBINE ----------
df["iso_anomaly_score"] = iso_scores
df["iso_flag"] = iso_pred
df["combined_anomaly"] = ((df["iso_flag"] == 1) |
                          (df["zscore_flag"] == 1)).astype(int)

# ---------- SAVE ----------
joblib.dump(iso, f"{MODELS}/isolation_forest.pkl")
df[["account_id", "iso_anomaly_score", "iso_flag",
    "zscore_anomaly", "zscore_flag", "combined_anomaly"]].to_csv(
    f"{PROCESSED}/anomaly_scores.csv", index=False)

json.dump({
    "iso_roc_auc": float(roc_auc_score(y, iso_scores)),
    "zscore_threshold": float(thr),
}, open(f"{OUT}/anomaly_metrics.json", "w"), indent=2)

print("\nAnomaly detection complete.")