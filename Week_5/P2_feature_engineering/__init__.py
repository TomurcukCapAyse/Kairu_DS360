"""
P2: Feature Engineering Module
Bu modül time series için feature engineering işlemlerini içerir.

Modüller:
- feature_engineering.py: Lag, rolling ve seasonal features

Usage:
    from P2_feature_engineering import create_features
    
    # Feature engineering çalıştır
    fe_train, fe_valid, X_train, y_train, X_valid, y_valid = create_features()
    
    # Veya sadece başarı kontrolü
    from P2_feature_engineering import main
    success = main()
"""

__version__ = "1.0.0"
__author__ = "M5 Forecasting Team"

try:
    from .feature_engineering import create_features, main
    
    __all__ = ['create_features', 'main']
    
except ImportError as e:
    print(f"⚠️ P2 Feature Engineering modülü yüklenemedi: {e}")
    __all__ = []