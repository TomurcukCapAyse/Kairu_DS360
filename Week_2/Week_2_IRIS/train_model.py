import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import json

def train_iris_model(model_type = 'random_forest'):
    """Iris veri seti üzerinde model eğitimi yap"""

    # Temizlenmiş veriyi yükle
    df = pd.read_csv('data/processed/iris_processed.csv')

    # Özellik ve hedef değişkeni ayır
    feature_cols = ['sepal_length', 'sepal_width', 
                    'petal_length', 'petal_width']
    
    X = df[feature_cols]
    y = df['species']

    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    # Modeli seç ve eğit
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                        random_state=42)
        
    else:
        model = LogisticRegression(random_state=42, max_iter=1000)

    # Modeli eğit
    model.fit(X_train, y_train)

    # Tahmin yap
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Performans metriklerini hesapla
    accuracy = accuracy_score(y_test, y_pred)

    # Model kaydetme dizinini oluştur
    os.makedirs('models', exist_ok=True)
    model_path = f'models/iris_{model_type}_model.pkl'
    joblib.dump(model, model_path)

    # Metrikleri kaydet
    metrics = {
        'model_type': model_type,
        'accuracy': float(accuracy),
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
    }

    with open('models/iris_model_metrics.json', 'w') as f:
        json.dump(feature_cols, f , indent=2)

    print(f"Model '{model_type}' eğitildi ve '{model_path}' dosyasına kaydedildi.")
    print(f"Accuracy: {accuracy:.4f}")
    # Detaylı sınıflandırma raporu
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, metrics

if __name__ == "__main__":
    # Random Forest modeli ile eğit
    model, metrics = train_iris_model(model_type='random_forest')

    # Lojistik Regresyon modeli ile eğit
    model_lr, metrics_lr = train_iris_model(model_type='logistic_regression')


    print("\n🎯 Her iki model de eğitildi ve kaydedildi!")