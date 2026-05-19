"""
train_model.py
--------------
Trains a Random Forest classifier on the cleaned diabetes dataset
and saves the full sklearn Pipeline (preprocessor + model) as model.pkl.

Run this once locally before deploying:
    python train_model.py
"""

import pickle
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ── 1. Load data ──────────────────────────────────────────────────────────────
DATA_PATH = "../Data/Processed/cleaned_df.pkl"

with open(DATA_PATH, "rb") as f:
    cleaned_df = pickle.load(f)

print(f"Dataset loaded: {cleaned_df.shape}")

# ── 2. Define features & target ───────────────────────────────────────────────
COLUMNS_TO_EXCLUDE = ["readmitted", "readmitted_30d", "cum_readmissions"]

X_full = cleaned_df.drop(columns=COLUMNS_TO_EXCLUDE, errors="ignore")
Y_binary = (cleaned_df["readmitted"] == "<30").astype(int)

# ── 3. Patient-level split (prevents data leakage) ───────────────────────────
unique_patients = X_full["patient_nbr"].unique()
patient_train_ids, patient_test_ids = train_test_split(
    unique_patients, test_size=0.2, random_state=42
)

X_train = X_full[X_full["patient_nbr"].isin(patient_train_ids)].copy()
y_train = Y_binary.loc[X_train.index].copy()
X_test  = X_full[X_full["patient_nbr"].isin(patient_test_ids)].copy()
y_test  = Y_binary.loc[X_test.index].copy()

# Drop patient ID — not a feature
X_train.drop("patient_nbr", axis=1, inplace=True)
X_test.drop("patient_nbr", axis=1, inplace=True)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Class balance (train): {y_train.value_counts().to_dict()}")

# ── 4. Preprocessing ──────────────────────────────────────────────────────────
numeric_cols     = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X_train.select_dtypes(include="object").columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"), categorical_cols),
])

# ── 5. Model pipeline ─────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=50,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model),
])

# ── 6. Train ──────────────────────────────────────────────────────────────────
print("Training model … (this may take a minute)")
pipe.fit(X_train, y_train)

# ── 7. Quick evaluation ───────────────────────────────────────────────────────
from sklearn.metrics import classification_report, roc_auc_score

y_pred  = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

print("\n── Evaluation ──────────────────────────────")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ── 8. Save pipeline + metadata ───────────────────────────────────────────────
joblib.dump(pipe, "model.pkl")
print("\nmodel.pkl saved ✓")

# Save feature column lists so the app can build the input DataFrame correctly
import json
feature_meta = {
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "feature_order": numeric_cols + categorical_cols,  # same order as X_train
}
with open("feature_meta.json", "w") as f:
    json.dump(feature_meta, f, indent=2)
print("feature_meta.json saved ✓")
