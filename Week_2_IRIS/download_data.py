import pandas as pd
import seaborn as sns
import os

def download_iris_data():

    # Veri dizini oluştur
    os.makedirs('data/raw', exist_ok=True)

    # Iris veri setini yükle
    df = sns.load_dataset('iris')

    # Veriyi CSV dosyasına kaydet
    df.to_csv('data/raw/iris.csv', index=False)

    print("Iris veri seti 'data/raw/iris.csv' dosyasına indirildi.")
    print(f"Veri seti boyutu: {df.shape}")
    print(f"Kolonlar: {list(df.columns)}")
    print(f"Eksik değerler:\n{df.isnull().sum()}")

    return df

if __name__ == "__main__":
    download_iris_data()