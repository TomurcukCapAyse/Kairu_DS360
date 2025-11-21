import pandas as pd
from pathlib import Path

RAW_PATH = Path(r"C:\Users\eyyupcap\Desktop\Ayşe\VS Code\Kairu_DS360\Week_4\fraud_detection\data\raw\creditcard_fraud.csv")
NEW_DATA_DIR = Path("new")
NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Ana veri
df = pd.read_csv(RAW_PATH)

# İlk batch: 1000 satır
df.sample(1000, random_state=42).to_csv(NEW_DATA_DIR / "new_data_day1.csv", index=False)

# İkinci batch: 2000 satır
df.sample(2000, random_state=43).to_csv(NEW_DATA_DIR / "new_data_day2.csv", index=False)

print("[OK] Yeni veri simülasyon dosyaları üretildi:", list(NEW_DATA_DIR.glob("*.csv")))