"""
============================================================
MuleWatch - Step 1: Synthetic Data Generation
============================================================
PURPOSE:
  Generate 7 synthetic security data sources that mimic a
  retail bank's fraud-relevant telemetry. No real data is used.

WHAT IT CREATES (in ../02_data/raw/):
  1. accounts.csv           - customer accounts + KYC + mule label
  2. devices.csv            - device fingerprints
  3. auth_logs.csv          - authentication / MFA events
  4. transactions.csv       - money movements
  5. beneficiary_changes.csv- payee additions
  6. geolocation.csv        - coarse location context
  7. aml_notes.csv          - unstructured AML case notes (NLP source)
  8. metadata.json          - generation parameters for reproducibility

HOW TO RUN:
  cd 03_notebooks_or_scripts
  python 01_data_generation.py
============================================================
"""

# ---------- IMPORTS ----------
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime
import random
import os
import json

# ---------- CONFIGURATION ----------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

N_ACCOUNTS = 5000            # total bank accounts
N_TRANSACTIONS = 75000       # total money movements
N_AUTH_EVENTS = 20000        # total login/MFA events
N_DEVICES = 8000             # total device fingerprints
N_BENEFICIARY_CHANGES = 6000 # total payee changes
N_AML_NOTES = 1000           # total free-text case notes
FRAUD_RATE = 0.03            # 3% of accounts are mule/fraud

OUTPUT_DIR = "../02_data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. ACCOUNTS (KYC base population + mule label)
# ============================================================
def generate_accounts():
    """Create the base account population with KYC attributes
    and a ground-truth 'is_mule' label for supervised training."""
    print("[1/7] Generating accounts...")
    rows = []
    for i in range(N_ACCOUNTS):
        rows.append({
            "account_id": f"ACC{i:06d}",
            "onboarding_date": fake.date_between(start_date="-5y", end_date="today"),
            "occupation_category": random.choice(
                ["salaried", "self_employed", "student", "retired", "business"]),
            "risk_rating": random.choices(
                ["low", "medium", "high"], weights=[0.7, 0.25, 0.05])[0],
            "account_type": random.choice(["savings", "current", "business"]),
        })
    df = pd.DataFrame(rows)

    # Inject labelled mule accounts
    fraud_idx = np.random.choice(df.index, size=int(N_ACCOUNTS * FRAUD_RATE), replace=False)
    df["is_mule"] = 0
    df.loc[fraud_idx, "is_mule"] = 1

    df.to_csv(f"{OUTPUT_DIR}/accounts.csv", index=False)
    print(f"    -> {len(df)} accounts ({df['is_mule'].sum()} mule)")
    return df

# ============================================================
# 2. DEVICES (device fingerprints)
# ============================================================
def generate_devices():
    """Generate device identifiers used in authentication events."""
    print("[2/7] Generating devices...")
    rows = []
    for i in range(N_DEVICES):
        rows.append({
            "device_id": f"DEV{i:06d}",
            "device_hash": f"hash_{random.getrandbits(64):016x}",
            "os": random.choice(["Android", "iOS", "Windows", "macOS", "Linux"]),
            "browser": random.choice(["Chrome", "Firefox", "Safari", "Edge"]),
            "first_seen": fake.date_time_between(start_date="-2y", end_date="now"),
            "last_seen": fake.date_time_between(start_date="-30d", end_date="now"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/devices.csv", index=False)
    print(f"    -> {len(df)} devices")
    return df

# ============================================================
# 3. AUTHENTICATION / MFA LOGS
# ============================================================
def generate_auth_logs(accounts, devices):
    """Create login/MFA events. Mule accounts show higher failure
    rates and off-hours activity."""
    print("[3/7] Generating auth/MFA logs...")
    mule_ids = set(accounts[accounts["is_mule"] == 1]["account_id"])
    all_ids = accounts["account_id"].tolist()
    dev_ids = devices["device_id"].tolist()

    rows = []
    for i in range(N_AUTH_EVENTS):
        acc = random.choice(all_ids)
        dev = random.choice(dev_ids)
        ts = fake.date_time_between(start_date="-6m", end_date="now")

        if acc in mule_ids and random.random() < 0.4:
            # Mule signature: failed MFA at odd hours
            mfa = random.choice(["fail", "challenge", "fail"])
            ts = ts.replace(hour=random.choice([1, 2, 3, 4, 23]))
        else:
            mfa = random.choices(["success", "fail", "challenge"],
                                 weights=[0.85, 0.10, 0.05])[0]
            ts = ts.replace(hour=random.randint(6, 22))

        rows.append({
            "event_id": f"AUTH{i:07d}",
            "account_id": acc,
            "device_id": dev,
            "timestamp": ts,
            "ip_address": fake.ipv4(),
            "mfa_result": mfa,
            "geo_city": fake.city(),
            "geo_country": fake.country_code(),
        })
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df.to_csv(f"{OUTPUT_DIR}/auth_logs.csv", index=False)
    print(f"    -> {len(df)} auth events")
    return df

# ============================================================
# 4. TRANSACTIONS
# ============================================================
def generate_transactions(accounts):
    """Create money movements. Mule accounts use sub-threshold
    amounts (structuring) and rapid pass-through transfers."""
    print("[4/7] Generating transactions...")
    mule_ids = set(accounts[accounts["is_mule"] == 1]["account_id"])
    all_ids = accounts["account_id"].tolist()

    rows = []
    for i in range(N_TRANSACTIONS):
        acc = random.choice(all_ids)
        cp = random.choice(all_ids)
        while cp == acc:
            cp = random.choice(all_ids)
        ts = fake.date_time_between(start_date="-6m", end_date="now")

        if acc in mule_ids:
            # Structuring: amounts just below NAD 10k reporting threshold
            amount = random.choice([
                round(random.uniform(9000, 9999), 2),
                round(random.uniform(4500, 4999), 2),
                round(random.uniform(100, 500), 2),
            ])
            channel = random.choice(["mobile", "internet"])
            ttype = random.choice(["transfer", "transfer", "withdrawal"])
        else:
            amount = min(round(np.random.lognormal(6, 1.5), 2), 50000)
            channel = random.choice(["mobile", "internet", "atm", "branch"])
            ttype = random.choice(["transfer", "payment", "withdrawal", "deposit"])

        rows.append({
            "txn_id": f"TXN{i:08d}",
            "account_id": acc,
            "counterparty_id": cp,
            "timestamp": ts,
            "amount": amount,
            "channel": channel,
            "txn_type": ttype,
            "currency": "NAD",
        })
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df.to_csv(f"{OUTPUT_DIR}/transactions.csv", index=False)
    print(f"    -> {len(df)} transactions")
    return df

# ============================================================
# 5. BENEFICIARY CHANGES
# ============================================================
def generate_beneficiary_changes(accounts):
    """Payee addition events. Mule accounts add beneficiaries
    shortly before rapid outbound transfers."""
    print("[5/7] Generating beneficiary changes...")
    rows = []
    for i in range(N_BENEFICIARY_CHANGES):
        rows.append({
            "change_id": f"BCH{i:06d}",
            "account_id": random.choice(accounts["account_id"].tolist()),
            "old_beneficiary": f"BEN{random.randint(10000, 99999)}",
            "new_beneficiary": f"BEN{random.randint(10000, 99999)}",
            "change_ts": fake.date_time_between(start_date="-6m", end_date="now"),
            "channel": random.choice(["mobile", "internet", "branch"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/beneficiary_changes.csv", index=False)
    print(f"    -> {len(df)} beneficiary changes")
    return df

# ============================================================
# 6. GEOLOCATION
# ============================================================
def generate_geolocation(auth_logs):
    """Derive coarse location context from auth events and flag
    'distance from home' anomalies."""
    print("[6/7] Generating geolocation context...")
    geo = auth_logs[["account_id", "timestamp", "geo_city", "geo_country"]].copy()
    geo["distance_from_home_flag"] = np.random.choice([0, 1], size=len(geo), p=[0.9, 0.1])
    geo.to_csv(f"{OUTPUT_DIR}/geolocation.csv", index=False)
    print(f"    -> {len(geo)} geolocation records")
    return geo

# ============================================================
# 7. AML CASE NOTES (unstructured text for NLP)
# ============================================================
def generate_aml_notes(accounts):
    """Free-text investigator notes with typology labels.
    Used for the NLP/text-mining component (Step 6)."""
    print("[7/7] Generating AML case notes...")
    templates = {
        "structuring": [
            "Multiple cash deposits just below the {threshold} threshold over {days} days.",
            "Customer made {count} transfers of NAD {amount} each, consistent with structuring.",
        ],
        "mule_network": [
            "Account linked to {count} other accounts sharing device {device}.",
            "Funds received from {count} unrelated parties and transferred out within minutes.",
        ],
        "account_takeover": [
            "Login from {city} followed by password reset and beneficiary change.",
            "MFA challenged {count} times within {minutes} minutes from new device.",
        ],
        "legitimate": [
            "Customer confirmed transaction via branch visit.",
            "Salary deposit and routine bill payments observed.",
        ],
    }
    labels = list(templates.keys())
    mule_ids = accounts[accounts["is_mule"] == 1]["account_id"].tolist()
    all_ids = accounts["account_id"].tolist()

    rows = []
    for i in range(N_AML_NOTES):
        if random.random() < 0.6:
            acc = random.choice(mule_ids)
            lbl = random.choice(["structuring", "mule_network", "account_takeover"])
        else:
            acc = random.choice(all_ids)
            lbl = random.choices(labels, weights=[0.15, 0.10, 0.15, 0.60])[0]

        text = random.choice(templates[lbl]).format(
            threshold=random.choice(["NAD 10,000", "NAD 25,000", "NAD 50,000"]),
            days=random.randint(2, 14),
            count=random.randint(2, 12),
            amount=random.randint(5000, 95000),
            device=f"DEV{random.randint(0, 9999):06d}",
            city=fake.city(),
            minutes=random.randint(5, 60),
        )
        rows.append({
            "case_id": f"CASE{i:05d}",
            "account_id": acc,
            "free_text_note": text,
            "typology_label": lbl,
            "created_at": fake.date_time_between(start_date="-6m", end_date="now"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/aml_notes.csv", index=False)
    print(f"    -> {len(df)} AML notes")
    return df

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("MuleWatch Synthetic Data Generation")
    print("=" * 55)
    accounts = generate_accounts()
    devices = generate_devices()
    auth = generate_auth_logs(accounts, devices)
    txns = generate_transactions(accounts)
    benef = generate_beneficiary_changes(accounts)
    geo = generate_geolocation(auth)
    notes = generate_aml_notes(accounts)

    # Save metadata for reproducibility
    meta = {
        "seed": SEED,
        "fraud_rate": FRAUD_RATE,
        "n_accounts": len(accounts),
        "n_transactions": len(txns),
        "n_auth_events": len(auth),
        "n_devices": len(devices),
        "n_beneficiary_changes": len(benef),
        "n_aml_notes": len(notes),
        "generated_at": datetime.now().isoformat(),
    }
    with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 55)
    print(f"DONE. Files written to {OUTPUT_DIR}")