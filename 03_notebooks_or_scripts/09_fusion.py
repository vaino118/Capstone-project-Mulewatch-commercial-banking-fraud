"""
============================================================
MuleWatch - Step 9: Score Fusion + SHAP Explainability
============================================================
PURPOSE:
  Combine supervised, anomaly, graph, and NLP scores into
  one fused risk score per account. Compute SHAP values
  for explainability. Produce the ranked alert queue.
INPUT:
  ../02_data/processed/account_features.csv
  ../02_data/processed/anomaly_scores.csv
  ../02_data/processed/graph_communities.csv
  ../02_data/processed/nlp_typology_scores.csv
  ../04_models/random_forest.pkl
OUTPUT:
  ../08_outputs/fused_alerts.csv
  ../08_outputs/shap_values.csv
HOW TO RUN:
  cd 03_notebooks_or_scripts
  python 09_fusion.py
============================================================
"""

import pandas as pd
import numpy as np
import joblib
import shap
import os

# ---------- PATHS ----------
PROCESSED = "../02_data/processed"
MODELS = "../04_models"
OUT = "../08_outputs"
os.makedirs(OUT, exist_ok=True)

# ---------- LOAD ALL SOURCES ----------
print("Loading all scores...")
feat = pd.read_csv(f"{PROCESSED}/account_features.csv")
anom = pd.read_csv(f"{PROCESSED}/anomaly_scores.csv")
graph = pd.read_csv(f"{PROCESSED}/graph_communities.csv")
nlp = pd.read_csv(f"{PROCESSED}/nlp_typology_scores.csv")
rf = joblib.load(f"{MODELS}/random_forest.pkl")

print(f"Features: {feat.shape}")
print(f"Anomaly:  {anom.shape}")
print(f"Graph:    {graph.shape}")
print(f"NLP:      {nlp.shape}")

# ---------- SUPERVISED PROBABILITY ----------
cols = [c for c in feat.columns if c not in ["account_id", "is_mule"]]
X = feat[cols].values
feat["p_mule"] = rf.predict_proba(X)[:, 1]

# ---------- MERGE ALL SCORES ----------
df = feat.merge(
    anom[["account_id", "iso_anomaly_score"]],
    on="account_id", how="left"
)
df = df.merge(
    graph[["account_id", "community", "graph_degree", "graph_betweenness"]],
    on="account_id", how="left"
)
df = df.merge(
    nlp[["account_id", "note_risk"]],
    on="account_id", how="left"
)

# ---------- NORMALISE EACH SCORE TO [0,1] ----------
def norm(s):
    """Min-max normalisation, safe against zero-range."""
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

df["p_mule_n"] = norm(df["p_mule"])
df["anomaly_n"] = norm(df["iso_anomaly_score"])
df["note_risk_n"] = norm(df["note_risk"].fillna(0))

# ---------- CLUSTER RISK ----------
# Compute mule density per Louvain community, then map back to accounts
comm_density = df.groupby("community")["is_mule"].mean().to_dict()
df["cluster_risk"] = df["community"].map(comm_density).fillna(0)
df["cluster_risk_n"] = norm(df["cluster_risk"])

# ---------- FUSION (weighted sum) ----------
W_SUP = 0.40
W_ANO = 0.20
W_GRP = 0.25
W_NLP = 0.15

df["risk_score"] = (
    W_SUP * df["p_mule_n"] +
    W_ANO * df["anomaly_n"] +
    W_GRP * df["cluster_risk_n"] +
    W_NLP * df["note_risk_n"]
)

df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)

# ---------- SHAP EXPLANATIONS ----------
# Compute on the top 500 accounts for speed.
# SHAP return type varies by version and model:
#   - list of 2 arrays        -> take [1]
#   - 3D ndarray (n,f,2)      -> take [:,:,1]
#   - 2D ndarray (n,f)        -> use as-is
print("\nComputing SHAP values (top 500 accounts)...")
explainer = shap.TreeExplainer(rf)
sample_X = X[:500]
shap_vals = explainer.shap_values(sample_X)

# Normalise to 2D
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]
elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
    shap_vals = shap_vals[:, :, 1]

print(f"SHAP shape: {shap_vals.shape}")

shap_df = pd.DataFrame(shap_vals, columns=cols)
shap_df.insert(0, "account_id", feat["account_id"].iloc[:500].values)
shap_df.to_csv(f"{OUT}/shap_values.csv", index=False)

# ---------- SAVE FUSED ALERTS ----------
df.to_csv(f"{OUT}/fused_alerts.csv", index=False)

print(f"\nFused alerts: {len(df)} accounts")
print(f"Top score: {df['risk_score'].iloc[0]:.4f}")
print(f"Alerts >= 0.60: {(df['risk_score'] >= 0.60).sum()}")
print(f"Alerts >= 0.85: {(df['risk_score'] >= 0.85).sum()}")
print(f"\nSaved: {OUT}/fused_alerts.csv")
print(f"Saved: {OUT}/shap_values.csv")
print("Fusion complete.")