import pandas as pd
import seaborn as sns
import os

def download_titanic_data():
    """Seaborn'dan Titanic datasetini indir."""

    # Veri dizinlerini oluştur
    os.makedirs('data/raw', exist_ok=True)

    # Titanic datasetini seaborn'dan yükle
    df = sns.load_dataset('titanic')
    
    #Ham veriyi CSV olarak kaydet
    df.to_csv('data/raw/titanic.csv', index=False)

    print("✅ Titanic verisi indirildi ve 'data/raw/titanic.csv' olarak kaydedildi.")
    print(f"Veri boyutu: {df.shape}")
    print(f"Kolonlar: {list(df.columns)}")
    print(f"Eksik değerler: \n{df.isnull().sum()}")

    return df

if __name__ == "__main__":
    download_titanic_data()