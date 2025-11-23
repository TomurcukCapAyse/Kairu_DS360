"""
P5: Machine Learning Models Module
Bu modül ML-based time series forecasting modellerini içerir.

Modüller:
- lightgbm_multi_item.py: LightGBM multi-item forecasting

Usage:
    from P5_ml_models import run_lightgbm
    # veya
    from P5_ml_models.lightgbm_multi_item import LightGBMMultiItemForecaster
"""

__version__ = "1.0.0"
__author__ = "M5 Forecasting Team"

try:
    from .lightgbm_multi_item import main as run_lightgbm
    from .lightgbm_multi_item import LightGBMMultiItemForecaster
    
    __all__ = ['run_lightgbm', 'LightGBMMultiItemForecaster']
    
except ImportError as e:
    # Import hatası durumunda boş liste
    __all__ = []
    print(f"⚠️  P5_ml_models import warning: {e}")