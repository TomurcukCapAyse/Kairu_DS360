# 🔍 Fraud Detection Pipeline

Kredi kartı dolandırıcılığı tespiti için end-to-end makine öğrenmesi pipeline'ı. Bu proje, veri ön işleme, outlier detection, model eğitimi, değerlendirme ve açıklanabilirlik modüllerini içeren kapsamlı bir MLOps çözümü sunar.

## 📊 Pipeline Çıktısı

```
INFO: Fraud Detection Pipeline initialized
INFO: Full Fraud Detection Pipeline başlatılıyor...
INFO: Preprocess: running FeaturePreprocessor
INFO: Training RandomForest (with subsample if needed)
INFO: Training set size 199364 > 100000, performing stratified subsample
INFO: Evaluating models (minimal)
INFO: Best model: random_forest (ROC-AUC: 0.9177)
INFO: Explain model: random_forest (stub)
INFO: Saving models (joblib)
INFO: Full pipeline completed successfully!
```

## 📋 İçindekiler

- #-özellikler
- #-dataset
- #-kurulum
- #-hızlı-başlangıç
- #-pipeline-bileşenleri
- #-kullanım
- #-konfigürasyon
- #-cicd-pipeline
- #-proje-yapısı
- #-sonuçlar

## ✨ Özellikler

| Modül | Özellikler |
|-------|------------|
| **Veri İndirme** | KaggleHub entegrasyonu, otomatik veri hazırlama |
| **Ön İşleme** | RobustScaler, OneHotEncoder, SMOTE/ADASYN desteği |
| **Outlier Detection** | Isolation Forest, Local Outlier Factor (LOF) |
| **Model Eğitimi** | Random Forest, stratified subsampling |
| **Değerlendirme** | ROC-AUC, PR-AUC, F1-Score, Confusion Matrix |
| **Açıklanabilirlik** | SHAP, LIME, Permutation Importance |
| **MLOps** | MLflow tracking, CI/CD pipeline, model versioning |

## 📁 Dataset

### Credit Card Fraud Detection (Kaggle)

| Özellik | Değer |
|---------|-------|
| **Toplam İşlem** | 284,807 |
| **Normal İşlem** | 284,315 (%99.83) |
| **Fraud İşlem** | 492 (%0.17) |
| **Features** | 30 (V1-V28 + Time + Amount) |
| **Eksik Değer** | 0 |

**Feature Açıklamaları:**
- `V1-V28`: PCA ile dönüştürülmüş gizli özellikler (gizlilik için)
- `Time`: İlk işlemden itibaren geçen süre (saniye)
- `Amount`: İşlem tutarı
- `Class`: 0 = Normal, 1 = Fraud

## 🚀 Kurulum

### Gereksinimler

- Python 3.9+
- pip

### Adımlar

```bash
# 1. Repository'yi klonlayın
git clone https://github.com/TomurcukCapAyse/Kairu_DS360.git
cd Kairu_DS360/Week_4/fraud_detection

# 2. Virtual environment oluşturun
python -m venv .venv

# 3. Aktifleştirin
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### Bağımlılıklar

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0,<1.4.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.15.0
shap>=0.45.0
lime>=0.2.0.1
imbalanced-learn>=0.11.0
mlflow>=2.8.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
joblib>=1.3.0
pyyaml>=6.0
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
kagglehub>=0.2.0
```

## ⚡ Hızlı Başlangıç

### 1. Dataset İndirme

```bash
python fraud_detection/download_data.py
```

**Çıktı:**
```
Credit Card Fraud Detection Dataset Download
============================================================
✅ Credit Card Fraud dataset hazır!
📁 Dosya konumu: fraud_detection/data/raw/creditcard_fraud.csv
📊 Dataset boyutu: ~150MB
📈 Dataset Özeti:
   Satır sayısı: 284,807
   Normal işlem: 284,315 (%99.83)
   Fraud işlem: 492 (%0.17)
```

### 2. Full Pipeline Çalıştırma

```bash
# Varsayılan (processed data ile)
python fraud_detection/src/pipeline.py

# Gerçek data ile
python fraud_detection/src/pipeline.py --data fraud_detection/data/raw/creditcard_fraud.csv --save_models

# KaggleHub ile otomatik indirme
python fraud_detection/src/pipeline.py --use_kagglehub --save_models
```

## 🔧 Pipeline Bileşenleri

### 1. Data Loading (`load_data`)

Pipeline otomatik olarak veriyi yükler ve train/test split yapar:

```python
from pipeline import FraudDetectionPipeline

pipeline = FraudDetectionPipeline()
pipeline.load_data(data_path="data/raw/creditcard_fraud.csv")
# veya synthetic=True ile demo data
```

**Özellikler:**
- Stratified train/test split (%80/%20)
- Processed data desteği
- KaggleHub entegrasyonu

### 2. Preprocessing (`preprocessing.py`)

```python
from preprocessing import FeaturePreprocessor, ImbalanceHandler

# Feature preprocessing
preprocessor = FeaturePreprocessor(
    scaling_method='robust',    # outlier'lara dayanıklı
    encoding_method='onehot'    # kategorik değişkenler için
)
df_processed = preprocessor.fit_transform(df, target_col='Class')

# Imbalance handling
X_balanced, y_balanced = ImbalanceHandler.apply_smote(X, y)
```

**Scaling Yöntemleri:**
| Yöntem | Kullanım Alanı |
|--------|----------------|
| `robust` | Outlier içeren veriler (önerilen) |
| `standard` | Normal dağılımlı veriler |
| `minmax` | Belirli aralık gerektiren durumlar |

**Imbalance Yöntemleri:**
- SMOTE: Sentetik minority oversampling
- ADASYN: Adaptive synthetic sampling
- SMOTETomek: SMOTE + Tomek links
- RandomUnderSampler: Majority undersampling

### 3. Outlier Detection (`outlier_detection.py`)

```python
# Isolation Forest & LOF ile anomaly detection
python fraud_detection/src/outlier_detection.py
```

**Çıktı:**
```
[IF]  ROC-AUC=0.9480 | PR-AUC=0.1381 | F1=0.261
[LOF] ROC-AUC=0.9320 | PR-AUC=0.1250 | F1=0.245
```

**Kullanım Stratejisi:**
- Outlier score'ları supervised modele ek feature olarak eklenebilir
- Threshold, F1 skorunu maximize eden noktadan seçilir
- Yüksek skor = daha anomali (fraud olasılığı yüksek)

### 4. Model Training (`train_models`)

```python
pipeline.train_models()
```

**Özellikler:**
- Random Forest Classifier
- Stratified subsampling (büyük veri setleri için)
- Configurable: `n_estimators`, `max_train_samples`

**Büyük Veri Yönetimi:**
```
Training set size 199364 > 100000, performing stratified subsample
```
- 100K'dan büyük veri setlerinde otomatik stratified subsample
- Class balance korunur

### 5. Evaluation (`evaluation.py`)

```python
from evaluation import FraudEvaluator

evaluator = FraudEvaluator(model=model, model_name="random_forest")
results = evaluator.evaluate_binary_classification(X_test, y_test, y_pred_proba=probs)

print(f"ROC-AUC: {results['roc_auc']:.4f}")
print(f"PR-AUC: {results['pr_auc']:.4f}")
print(f"F1-Score: {results['f1_score']:.4f}")
```

**Metrikler:**
| Metrik | Açıklama | Önem |
|--------|----------|------|
| ROC-AUC | Overall ayırt etme yeteneği | Genel performans |
| PR-AUC | Imbalanced data performansı | **Kritik** |
| Precision | Fraud dediğinde doğruluk | False alarm kontrolü |
| Recall | Gerçek fraud yakalama oranı | Fraud kaçırma riski |
| F1-Score | Precision-Recall dengesi | Trade-off |

### 6. Explainability (`explainability_clean.py`)

```python
from explainability_clean import ModelExplainer

explainer = ModelExplainer(
    model=model,
    X_train=X_train,
    feature_names=feature_names,
    class_names=['Normal', 'Fraud']
)

# SHAP Analysis
explainer.initialize_shap(explainer_type='tree')
shap_values, X_sample = explainer.compute_shap_values(X_test)
explainer.plot_shap_summary(X_sample)

# LIME Analysis
explainer.initialize_lime()
explainer.explain_instance_lime(X_test, instance_idx=0)

# Permutation Importance
explainer.compute_permutation_importance(X_test, y_test)
```

**Açıklanabilirlik Yöntemleri:**
| Yöntem | Tip | Kullanım |
|--------|-----|----------|
| SHAP | Global + Local | Feature importance, dependence plots |
| LIME | Local | Tek işlem açıklaması |
| Permutation | Global | Model-agnostic importance |

## 💻 CLI Kullanımı

```bash
python fraud_detection/src/pipeline.py 


| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `--config` | Config dosyası yolu | `config/config.yaml` |
| `--data` | Veri dosyası yolu | Processed/synthetic data |
| `--mode` | `train`, `predict`, `explain` | `train` |
| `--model` | Model adı | `random_forest` |
| `--load_models` | Mevcut modelleri yükle | `False` |
| `--save_models` | Modelleri kaydet | `False` |
| `--use_kagglehub` | KaggleHub ile veri indir | `False` |

### Örnek Kullanımlar

```bash
# Full training pipeline
python fraud_detection/src/pipeline.py --mode train --save_models

# Prediction with saved model
python fraud_detection/src/pipeline.py --mode predict --load_models

# Model explanation
python fraud_detection/src/pipeline.py --mode explain --load_models --model random_forest

# KaggleHub ile tam pipeline
python fraud_detection/src/pipeline.py --use_kagglehub --save_models
```

## ⚙️ Konfigürasyon

### config/config.yaml

```yaml
# Data Configuration
data:
  test_size: 0.3
  random_state: 42
  stratify: true

# Preprocessing
preprocessing:
  scaling_method: "robust"
  encoding_method: "onehot"

# Model Configuration
models:
  random_forest:
    n_estimators: 100
    max_depth: 10
    class_weight: "balanced"
  
  isolation_forest:
    contamination: 0.05
    n_estimators: 200

# Explainability
explainability:
  shap:
    explainer_type: "tree"
    max_samples: 100
  lime:
    num_features: 10

# MLflow
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "fraud_detection"
```

## 🔄 CI/CD Pipeline

GitHub Actions ile otomatik CI/CD:

```
┌─────────────────┐
│ Data Validation │ → Schema, quality checks
└────────┬────────┘
         │
┌────────▼────────┐
│  Code Quality   │ → Linting, formatting
└────────┬────────┘
         │
┌────────▼────────┐
│  Model Training │ → Automated training
└────────┬────────┘
         │
┌────────▼────────┐
│   Performance   │ → ROC-AUC, latency tests
└────────┬────────┘
         │
┌────────▼────────┐
│   Deployment    │ → Staging → Production
└────────┬────────┘
         │
┌────────▼────────┐
│   Monitoring    │ → Drift detection, alerts
└─────────────────┘
```

**Trigger:**
- `push` to main/develop
- `pull_request` to main

## 📁 Proje Yapısı

```
fraud_detection/
├── config/
│   └── config.yaml              # Pipeline konfigürasyonu
├── data/
│   ├── raw/                     # Ham veri (creditcard_fraud.csv)
│   └── processed/               # İşlenmiş veri
│       ├── train_processed_supervised.csv
│       ├── test_processed_supervised.csv
│       ├── dataset_with_anomaly_scores_raw.csv
│       ├── outlier_meta_raw.json
│       ├── dataset_processed_supervised.csv
│       └── preprocessing_comparison.png
├── models/                      # Kaydedilmiş modeller
│   └── random_forest.joblib
├── src/
│   ├── pipeline.py              # Ana pipeline
│   ├── preprocessing.py         # Feature preprocessing
│   ├── evaluation.py            # Model değerlendirme
│   ├── explainability_clean.py  # SHAP/LIME açıklanabilirlik
│   ├── outlier_detection.py     # IF/LOF anomaly detection
│   └── download_data.py         # Dataset indirme utility
├── .github/workflows/
│   ├── ci_cd.yml                # GitHub Actions CI/CD
│   └── config.yaml              # Fraud Detection Configuration
├── requirements.txt             # Python bağımlılıkları
└── README.md
```

## 📈 Sonuçlar

### Model Performansı

| Model | ROC-AUC | PR-AUC | F1-Score |
|-------|---------|--------|----------|
| **Random Forest** | **0.9177** | - | - |
| Isolation Forest | 0.9480 | 0.1381 | 0.261 |
| LOF | 0.9320 | 0.1250 | 0.245 |

### Outlier Detection Analizi

**Isolation Forest:**
- ROC-AUC yüksek → iyi anomaly score üretiyor
- PR-AUC düşük → doğrudan alarm mekanizması zayıf
- **Öneri:** `if_score`'u supervised modele ek feature olarak ekle

**LOF:**
- Density-based yaklaşım
- Lokal anomalileri yakalamada etkili

### Önerilen Kullanım

1. **Outlier score'ları feature olarak ekle** → Supervised model performansını artırır
2. **Threshold optimizasyonu** → Business cost'a göre ayarla
3. **Ensemble yaklaşım** → IF + LOF + Supervised model kombinasyonu


**🎯 Başlamak için:** `python fraud_detection/src/pipeline.py --save_models`