"""
============================================================
MuleWatch - Step 6: NLP Text Classifier
============================================================
PURPOSE: Classify AML case notes into typologies
         (structuring / mule_network / account_takeover /
          legitimate) using TF-IDF + Logistic Regression.
INPUT:   ../02_data/raw/aml_notes.csv
OUTPUT:  ../04_models/nlp_classifier.pkl
         ../02_data/processed/nlp_typology_scores.csv
         ../08_outputs/nlp_metrics.json
         ../08_outputs/aml_notes_scored.csv
============================================================
"""

import pandas as pd, numpy as np, joblib, os, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ---------- PATHS ----------
RAW = "../02_data/raw"
PROCESSED = "../02_data/processed"
MODELS = "../04_models"; os.makedirs(MODELS, exist_ok=True)
OUT = "../08_outputs"; os.makedirs(OUT, exist_ok=True)

# ---------- LOAD ----------
print("Loading AML notes...")
notes = pd.read_csv(f"{RAW}/aml_notes.csv")
print(f"Loaded {len(notes)} notes")
print(f"Label distribution:\n{notes['typology_label'].value_counts()}\n")

# ---------- TRAIN/TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    notes["free_text_note"],
    notes["typology_label"],
    test_size=0.3,
    random_state=42,
    stratify=notes["typology_label"],
)

# ---------- PIPELINE: TF-IDF + LOGISTIC REGRESSION ----------
print("Training TF-IDF + Logistic Regression pipeline...")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_features=2000,
        stop_words="english",
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )),
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# ---------- EVALUATE ----------
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, zero_division=0))

# ---------- SCORE ALL NOTES ----------
probs = pipeline.predict_proba(notes["free_text_note"])
classes = pipeline.classes_

# Risk weights per typology (0 = legitimate, 1 = worst)
risk_weights = {
    "legitimate": 0.0,
    "structuring": 0.8,
    "mule_network": 1.0,
    "account_takeover": 0.9,
}
risk_per_note = np.array([risk_weights.get(c, 0) for c in classes])

notes["note_risk"] = probs @ risk_per_note
notes["predicted_typology"] = pipeline.predict(notes["free_text_note"])

# ---------- AGGREGATE PER ACCOUNT ----------
account_scores = notes.groupby("account_id").agg(
    note_risk=("note_risk", "max"),
    n_notes=("case_id", "count"),
).reset_index()

account_scores.to_csv(f"{PROCESSED}/nlp_typology_scores.csv", index=False)

# ---------- SAVE MODEL & OUTPUTS ----------
joblib.dump(pipeline, f"{MODELS}/nlp_classifier.pkl")
notes.to_csv(f"{OUT}/aml_notes_scored.csv", index=False)

json.dump({
    "accuracy": float(acc),
    "classes": list(classes),
}, open(f"{OUT}/nlp_metrics.json", "w"), indent=2)

print(f"\nAccounts with notes: {len(account_scores)}")
print("NLP classifier complete.")