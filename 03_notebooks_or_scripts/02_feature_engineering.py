"""
============================================================
MuleWatch - Step 2: Feature Engineering
============================================================
PURPOSE:
  Transform raw events into one row per account with
  behavioural, velocity, beneficiary, and graph features.

INPUT:
  ../02_data/raw/*.csv  (from Step 1)

OUTPUT:
  ../02_data/processed/account_features.csv

HOW TO RUN:
  cd 03_notebooks_or_scripts
  python 02_feature_engineering.py
============================================================
"""

import pandas as pd
import numpy as np
import networkx as nx
import os

# ---------- PATHS ----------
RAW = "../02_data/raw"
PROCESSED = "../02_data/processed"
os.makedirs(PROCESSED, exist_ok=True)

# ---------- LOAD ----------
print("Loading raw data...")
accounts = pd.read_csv(f"{RAW}/accounts.csv", parse_dates=["onboarding_date"])
auth = pd.read_csv(f"{RAW}/auth_logs.csv", parse_dates=["timestamp"])
txns = pd.read_csv(f"{RAW}/transactions.csv", parse_dates=["timestamp"])
benef = pd.read_csv(f"{RAW}/beneficiary_changes.csv", parse_dates=["change_ts"])

# ============================================================
# 1. ACCESS / AUTH FEATURES
# ============================================================
print("Access features...")
auth_feat = auth.groupby("account_id").agg(
    auth_total=("event_id", "count"),
    auth_fail_count=("mfa_result", lambda x: (x == "fail").sum()),
    auth_challenge_count=("mfa_result", lambda x: (x == "challenge").sum()),
    auth_unique_devices=("device_id", "nunique"),
    auth_unique_countries=("geo_country", "nunique"),
    auth_unique_cities=("geo_city", "nunique"),
).reset_index()

auth_feat["auth_fail_rate"] = auth_feat["auth_fail_count"] / auth_feat["auth_total"]
auth_feat["auth_challenge_rate"] = auth_feat["auth_challenge_count"] / auth_feat["auth_total"]

auth["hour"] = auth["timestamp"].dt.hour
auth["is_off"] = auth["hour"].apply(lambda h: 1 if h >= 22 or h < 6 else 0)
off = auth.groupby("account_id")["is_off"].mean().reset_index()
off.columns = ["account_id", "auth_off_hours_rate"]
auth_feat = auth_feat.merge(off, on="account_id", how="left")

# ============================================================
# 2. TRANSACTION FEATURES
# ============================================================
print("Transaction features...")
txn_feat = txns.groupby("account_id").agg(
    txn_total=("txn_id", "count"),
    txn_amount_mean=("amount", "mean"),
    txn_amount_std=("amount", "std"),
    txn_amount_max=("amount", "max"),
    txn_amount_sum=("amount", "sum"),
    txn_unique_counterparties=("counterparty_id", "nunique"),
).reset_index()
txn_feat["txn_amount_cv"] = txn_feat["txn_amount_std"] / txn_feat["txn_amount_mean"]

txns["is_sub"] = ((txns["amount"] >= 9000) & (txns["amount"] < 10000)).astype(int)
sub = txns.groupby("account_id")["is_sub"].mean().reset_index()
sub.columns = ["account_id", "txn_sub_threshold_ratio"]
txn_feat = txn_feat.merge(sub, on="account_id", how="left")

txns_s = txns.sort_values(["account_id", "timestamp"])
txns_s["prev_ts"] = txns_s.groupby("account_id")["timestamp"].shift(1)
txns_s["gap_min"] = (txns_s["timestamp"] - txns_s["prev_ts"]).dt.total_seconds() / 60
rapid = txns_s.groupby("account_id").apply(
    lambda g: (g["gap_min"] < 60).mean()
).reset_index()
rapid.columns = ["account_id", "txn_rapid_ratio"]
txn_feat = txn_feat.merge(rapid, on="account_id", how="left")

# ============================================================
# 3. BENEFICIARY FEATURES
# ============================================================
print("Beneficiary features...")
benef_feat = benef.groupby("account_id").agg(
    benef_change_count=("change_id", "count"),
    benef_unique_new=("new_beneficiary", "nunique"),
).reset_index()

# ============================================================
# 4. GRAPH FEATURES
# ============================================================
print("Graph features (may take ~1 min)...")
G = nx.Graph()
G.add_edges_from(zip(txns["account_id"], txns["counterparty_id"]))

deg = dict(G.degree())
btw = nx.betweenness_centrality(G, k=min(500, len(G)))
clu = nx.clustering(G)

graph_feat = pd.DataFrame({
    "account_id": list(G.nodes()),
    "graph_degree": [deg.get(n, 0) for n in G.nodes()],
    "graph_betweenness": [btw.get(n, 0) for n in G.nodes()],
    "graph_clustering": [clu.get(n, 0) for n in G.nodes()],
})

# ============================================================
# 5. MERGE
# ============================================================
print("Merging...")
feat = accounts[["account_id", "is_mule"]].copy()
for d in [auth_feat, txn_feat, benef_feat, graph_feat]:
    feat = feat.merge(d, on="account_id", how="left")
feat = feat.fillna(0)

feat.to_csv(f"{PROCESSED}/account_features.csv", index=False)
print(f"Saved: {feat.shape} -> 02_data/processed/account_features.csv")