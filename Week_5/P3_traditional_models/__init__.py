"""
P3: Traditional Models Module
Bu modül geleneksel istatistiksel time series modellerini içerir.

Modüller:
- arima_single_item.py: ARIMA model implementation

Usage:
    from P3_traditional_models import run_arima
    # veya
    from P3_traditional_models.arima_single_item import ARIMASingleItemForecaster
"""

__version__ = "1.0.0"
__author__ = "M5 Forecasting Team"

try:
    from .arima_single_item import main as run_arima
    from .arima_single_item import ARIMASingleItemForecaster
    
    __all__ = ['run_arima', 'ARIMASingleItemForecaster']
    
except ImportError as e:
    # Import hatası durumunda boş liste
    __all__ = []
    print(f"⚠️  P3_traditional_models import warning: {e}")