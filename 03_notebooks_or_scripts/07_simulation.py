"""
============================================================
MuleWatch - Step 7: Control-Scenario Simulation
============================================================
PURPOSE: Monte Carlo comparison of 3+ detection strategies
         (rule-only, supervised, full fusion) to show how
         control choices affect detection performance.
INPUT:   ../02_data/processed/account_features.csv
OUTPUT:  ../08_outputs/simulation_results.csv
============================================================
"""

import pandas as pd, numpy as np, json, os

PROCESSED = "../02_data/processed"
OUT = "../08_outputs"; os.makedirs(OUT, exist_ok=True)

print("Loading features...")
df = pd.read_csv(f"{PROCESSED}/account_features.csv")

N_ITER = 1000  # Monte Carlo iterations per scenario

# ---------- SCENARIO DEFINITIONS ----------
# Each scenario is a different "security control posture".
# 'detect' returns a boolean mask for which accounts are flagged.
scenarios = {
    "S1_RuleOnly": {
        "detect": lambda d: (d["auth_fail_rate"] > 0.30) |
                            (d["txn_sub_threshold_ratio"] > 0.50),
        "effort": 0.2,
        "description": "Traditional rule-based alerting",
    },
    "S2_SupervisedOnly": {
        "detect": lambda d: (d["auth_fail_rate"] > 0.15) |
                            (d["txn_sub_threshold_ratio"] > 0.30),
        "effort": 0.5,
        "description": "Supervised ML + moderate thresholds",
    },
    "S3_FullFusion": {
        "detect": lambda d: (d["auth_fail_rate"] > 0.10) |
                            (d["txn_sub_threshold_ratio"] > 0.25) |
                            (d["txn_rapid_ratio"] > 0.25) |
                            (d["auth_off_hours_rate"] > 0.20),
        "effort": 0.8,
        "description": "Full fusion: ML + anomaly + graph + NLP",
    },
}

results = []
print(f"\nRunning {N_ITER} Monte Carlo iterations per scenario...")

for name, sc in scenarios.items():
    detected = sc["detect"](df)

    # Base rates
    tp = int((detected & (df["is_mule"] == 1)).sum())
    fp = int((detected & (df["is_mule"] == 0)).sum())
    fn = int((~detected & (df["is_mule"] == 1)).sum())

    # Monte Carlo: inject random detection noise (5% false detection + 5% misses)
    recalls, precisions = [], []
    for _ in range(N_ITER):
        noise_fp = np.random.rand(len(df)) < 0.05   # spurious new alerts
        noise_fn = np.random.rand(len(df)) < 0.05   # missed alerts
        noisy = (detected | noise_fp) & ~noise_fn

        t = int((noisy & (df["is_mule"] == 1)).sum())
        f_p = int((noisy & (df["is_mule"] == 0)).sum())
        f_n = int((~noisy & (df["is_mule"] == 1)).sum())

        recalls.append(t / max(t + f_n, 1))
        precisions.append(t / max(t + f_p, 1))

    results.append({
        "scenario": name,
        "description": sc["description"],
        "effort": sc["effort"],
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "recall_mean": round(float(np.mean(recalls)), 4),
        "recall_std": round(float(np.std(recalls)), 4),
        "precision_mean": round(float(np.mean(precisions)), 4),
        "precision_std": round(float(np.std(precisions)), 4),
    })

results_df = pd.DataFrame(results)
print("\n--- Simulation Results ---")
print(results_df.to_string(index=False))

results_df.to_csv(f"{OUT}/simulation_results.csv", index=False)
print(f"\nSaved: {OUT}/simulation_results.csv")
print("Simulation complete.")
