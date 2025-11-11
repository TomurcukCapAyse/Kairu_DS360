import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def clean_iris_data(input_path='data/raw/iris.csv',
                     output_path='data/processed/iris_processed.csv'):
    """Veri setini temizle"""

    # Veriyi yükleme yapalım
    df = pd.read_csv(input_path)

    # Kopyasını alalaım
    df_cleaned = df.copy()

    # Eksik değer ve gereksiz kolon yok species hedef değişken
    # Species kolonunu encode edelim
    le_species = LabelEncoder()
    df_cleaned['species'] = le_species.fit_transform(df_cleaned['species'])

    # Çıktı dizini oluşturalım
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Temizlenmiş veriyi kaydedelim
    df_cleaned.to_csv(output_path, index=False)

    print(f"Temizlenmiş veri seti '{output_path}' dosyasına kaydedildi.")
    print(f"Orijinal veri seti boyutu: {df.shape}")
    print(f"Temizlenmiş veri seti boyutu: {df_cleaned.shape}")
    print(f"Kolonlar: {list(df_cleaned.columns)}")

    # Özellik listesini döndürelim
    features = df_cleaned.columns.tolist()

    print(f"Özellikler: {features}")

    return df_cleaned, features

if __name__ == "__main__":
    clean_iris_data()