import pandas as pd
from sqlalchemy import create_engine, text

# Bağlantı: DB adın farklıysa 'postgres' yerine onu yaz
engine = create_engine("postgresql://postgres:Tomurcuk.1998@localhost:5432/postgres")

# (İsteğe bağlı) emin olmak için aktif DB ve search_path’i göster
with engine.begin() as conn:
    print("DB  :", conn.execute(text("SELECT current_database();")).scalar())
    print("Path:", conn.execute(text("SHOW search_path;")).scalar())

# 1) Tüm tablo (büyük veri ise dikkat)
with engine.connect() as conn:
    df = pd.read_sql(text("SELECT * FROM public.creditcard_fraud;"), conn)

print(df.shape)
print(df.head())