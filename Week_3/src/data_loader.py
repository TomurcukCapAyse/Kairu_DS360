import pandas as pd
import kagglehub
import os

def download_loan_data():
    """
    Downloads the loan data from Kaggle and loads it into a pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the loan data.
    """
    # Download the dataset latest version using kagglehub
    path = kagglehub.dataset_download("zhijinzhai/loandata")
    print(f"Dataset downloaded to: {path}")
    return path

def load_data():
    """
    Loads the loan data from the downloaded CSV file into a pandas DataFrame.

    """

    # Data klasörünü oluştur - yolu belirle
    script_dir = os.path.dirname(os.path.abspath(__file__))  # src klasörü
    project_dir = os.path.dirname(script_dir)  # Week_3 klasörü
    data_dir = os.path.join(project_dir, 'data')
    local_file = os.path.join(data_dir, 'loan_data.csv')

    os.makedirs(data_dir, exist_ok=True)
    print(f"📁 Data klasörü: {data_dir}")
    print(f"📄 Aranan dosya: {local_file}")


    # Önce yerel dosyayı kontrol et
    if os.path.exists(local_file):
        df = pd.read_csv(local_file)
        print("✅ Data loaded from local file.")
    else:

        print("Kaggle'den veri indiriliyor...")
        dataset_path = download_loan_data()

        # Dataset dosyalarını kontrol et
        files = os.listdir(dataset_path)
        print(f"Downloaded files: {files}")

        # İlk CSV dosyasını bul ve yükle
        csv_files = [f for f in files if f.endswith('.csv')]
        if csv_files:
            df = pd.read_csv(os.path.join(dataset_path, csv_files[0]))

            # Verinin yerel kopyasını kaydet
            df.to_csv(local_file, index=False)
            print("Data loaded from downloaded file and saved locally.")
        else:
            raise Exception("No CSV file found in the downloaded dataset.")
        
    return df

if __name__ == "__main__":
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())