"""
=============================================================
  ONLINE PAYMENTS FRAUD DETECTION - Model Training Script
=============================================================
Run this file first to train the model and generate payments.pkl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
print("=" * 55)
print("  ONLINE PAYMENTS FRAUD DETECTION - TRAINING")
print("=" * 55)

csv_path = os.path.join(os.path.dirname(__file__),
                        '../data/fraud_dataset.csv')
print(f"\n[1/6] Loading dataset from:\n      {csv_path}")
df = pd.read_csv(csv_path)
print(f"      ✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. EXPLORE DATA
# ─────────────────────────────────────────────
print("\n[2/6] Exploring data...")
print(f"      Columns      : {df.columns.tolist()}")
print(f"      Missing vals : {df.isnull().sum().sum()}")
print(f"      Fraud cases  : {df['isFraud'].sum():,} "
      f"({df['isFraud'].mean()*100:.4f}%)")
print(f"      Trans types  : {df['type'].unique().tolist()}")

# ─────────────────────────────────────────────
# 3. PREPROCESS
# ─────────────────────────────────────────────
print("\n[3/6] Preprocessing data...")

# Encode transaction type
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])
# Mapping: CASH_IN=0, CASH_OUT=1, DEBIT=2, PAYMENT=3, TRANSFER=4

# Drop columns not needed for prediction
df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

# Balance dataset (fraud is very rare ~0.1%)
fraud_df    = df[df['isFraud'] == 1]
non_fraud_df = df[df['isFraud'] == 0].sample(n=30000, random_state=42)
balanced_df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42)

print(f"      Fraud samples     : {len(fraud_df):,}")
print(f"      Non-fraud samples : {len(non_fraud_df):,}")
print(f"      Balanced total    : {len(balanced_df):,}")

X = balanced_df.drop('isFraud', axis=1)
y = balanced_df['isFraud']

# ─────────────────────────────────────────────
# 4. SPLIT DATA
# ─────────────────────────────────────────────
print("\n[4/6] Splitting data (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"      Train : {X_train.shape[0]:,} samples")
print(f"      Test  : {X_test.shape[0]:,} samples")
print(f"      Features: {X_train.columns.tolist()}")

# ─────────────────────────────────────────────
# 5. TRAIN MODEL
# ─────────────────────────────────────────────
print("\n[5/6] Training Random Forest Classifier...")
print("      (This may take 1-2 minutes...)")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1          # use all CPU cores
)
model.fit(X_train, y_train)
print("      ✅ Model trained!")

# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
print("\n[6/6] Evaluating model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n  ┌─────────────────────────────────┐")
print(f"  │  Accuracy : {acc*100:6.2f}%              │")
print(f"  └─────────────────────────────────┘")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=['Legitimate', 'Fraud']))

print("  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"    True Negatives  : {cm[0][0]:,}")
print(f"    False Positives : {cm[0][1]:,}")
print(f"    False Negatives : {cm[1][0]:,}")
print(f"    True Positives  : {cm[1][1]:,}")

# Feature importance
print("\n  Feature Importances:")
fi = pd.Series(model.feature_importances_, index=X.columns)
for feat, imp in fi.sort_values(ascending=False).items():
    bar = "█" * int(imp * 50)
    print(f"    {feat:<20} {bar}  {imp:.4f}")

# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────
pkl_path = os.path.join(os.path.dirname(__file__), 'payments.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(model, f)

# Also copy to flask folder
flask_pkl = os.path.join(os.path.dirname(__file__), '../flask/payments.pkl')
with open(flask_pkl, 'wb') as f:
    pickle.dump(model, f)

print(f"\n  ✅ Model saved → training/payments.pkl")
print(f"  ✅ Model copied → flask/payments.pkl")
print("\n  👉 Now run:  cd ../flask && python app.py")
print("=" * 55)
