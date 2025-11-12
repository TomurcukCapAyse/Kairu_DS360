"""
EDA Script - Loan Risk
Aptala anlat: Veriyi oku , özetle , hedef dağılımına bak , leakage kolonlarını işaretle.
"""

import pandas as pd
import os

# Platform bağımsız yol
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
DATA_PATH = os.path.join(project_dir, 'data', 'loan_data.csv')

def perform_eda():
    df = pd.read_csv(DATA_PATH)
    print("📦 Shape:", df.shape)
    print("\n 🧱 dtypes:\n", df.dtypes)

    print("\n 🕳️ Eksikler:\n", df.isna().sum())

    if "loan_status" in df.columns:
        print("\n🎯 loan_status dağılımı:\n", df["loan_status"].value_counts(dropna=False))
    
    # Tarihleri parse et (EDA için)
    for c in ["effective_date","due_date","paid_off_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    
    # Basit türev: planlanan vade (gün)
    if {"effective_date","due_date"}.issubset(df.columns):
        df["planned_term_days"] = (df["due_date"] - df["effective_date"]).dt.days

    # Numerik özet
    num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    print("\n 🔢 Sayısal kolonlar:", num_cols)
    if num_cols:
        print(df[num_cols].describe())

    # Kategorik özet
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print("\n 🔢 Kategorik kolonlar:", cat_cols)
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False)
        print(f"\n {c} dağılımı: \n {vc}")

    # Leakage uyarısı
    print("\n⚠️ Leakage (eğitimde kullanma): paid_off_time, past_due_days")
    print("ℹ️ Kimlik kolonlarını (Loan_ID) da modele verme.")

if __name__ == "__main__":
    perform_eda()