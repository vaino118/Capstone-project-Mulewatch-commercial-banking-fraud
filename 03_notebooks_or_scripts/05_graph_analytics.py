"""
============================================================
MuleWatch - Step 5: Graph & Community Analytics
============================================================
PURPOSE: Build the account-to-account transaction graph,
         run Louvain community detection, compute centrality.
         Surfaces mule-account clusters.
INPUT:   ../02_data/raw/transactions.csv
         ../02_data/raw/accounts.csv
OUTPUT:  ../02_data/processed/graph_communities.csv
         ../08_outputs/txn_graph.pkl
         ../08_outputs/community_summary.csv
============================================================
"""

import pandas as pd, networkx as nx, community as louvain
import json, os, pickle

# ---------- PATHS ----------
RAW = "../02_data/raw"
PROCESSED = "../02_data/processed"
OUT = "../08_outputs"; os.makedirs(OUT, exist_ok=True)

# ---------- LOAD ----------
print("Loading transactions + accounts...")
txns = pd.read_csv(f"{RAW}/transactions.csv")
accounts = pd.read_csv(f"{RAW}/accounts.csv")

# ---------- BUILD GRAPH ----------
print("Building account graph...")
G = nx.Graph()
# Each transaction creates an edge between two accounts
G.add_edges_from(zip(txns["account_id"], txns["counterparty_id"]))
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ---------- LOUVAIN COMMUNITY DETECTION ----------
print("Running Louvain community detection...")
partition = louvain.best_partition(G, random_state=42)
communities = pd.Series(partition, name="community").reset_index()
communities.columns = ["account_id", "community"]

# ---------- CENTRALITY METRICS ----------
print("Computing centrality (may take ~30 sec)...")
deg = dict(G.degree())
# k=500 samples 500 nodes for betweenness estimation (faster)
btw = nx.betweenness_centrality(G, k=min(500, len(G)))

com_df = pd.DataFrame({
    "account_id": list(G.nodes()),
    "graph_degree": [deg.get(n, 0) for n in G.nodes()],
    "graph_betweenness": [btw.get(n, 0) for n in G.nodes()],
})
com_df = com_df.merge(communities, on="account_id", how="left")
com_df = com_df.merge(accounts[["account_id", "is_mule"]],
                       on="account_id", how="left")

# ---------- COMMUNITY SUMMARY ----------
summary = com_df.groupby("community").agg(
    size=("account_id", "count"),
    mule_count=("is_mule", "sum"),
).reset_index()
summary["mule_density"] = summary["mule_count"] / summary["size"]
summary = summary.sort_values("mule_density", ascending=False)

print("\nTop 10 suspicious communities:")
print(summary.head(10).to_string(index=False))

# ---------- SAVE ----------
summary.to_csv(f"{OUT}/community_summary.csv", index=False)
com_df.to_csv(f"{PROCESSED}/graph_communities.csv", index=False)

with open(f"{OUT}/txn_graph.pkl", "wb") as f:
    pickle.dump(G, f)

print(f"\nCommunities: {summary.shape[0]}")
print("Graph analytics complete.")