"""
MuleWatch - Score a single new account (deployment-style demo)
"""
import pandas as pd, joblib, numpy as np

MODELS = "../04_models"
rf = joblib.load(f"{MODELS}/random_forest.pkl")

# Simulate one new account arriving at the bank
new_account = {
    "auth_total": 12, "auth_fail_count": 5, "auth_challenge_count": 2,
    "auth_unique_devices": 3, "auth_unique_countries": 2,
    "auth_unique_cities": 2, "auth_fail_rate": 0.42,
    "auth_challenge_rate": 0.17, "auth_off_hours_rate": 0.58,
    "txn_total": 8, "txn_amount_mean": 9500, "txn_amount_std": 500,
    "txn_amount_max": 9990, "txn_amount_sum": 76000,
    "txn_unique_counterparties": 1, "txn_amount_cv": 0.05,
    "txn_sub_threshold_ratio": 0.88, "txn_rapid_ratio": 0.75,
    "benef_change_count": 3, "benef_unique_new": 3,
    "graph_degree": 12, "graph_betweenness": 0.15, "graph_clustering": 0.4,
}

df = pd.DataFrame([new_account])
prob = rf.predict_proba(df.values)[0][1]
print(f"Malicious probability: {prob:.4f}")
print(f"Prediction: {'MULE' if prob > 0.5 else 'legitimate'}")