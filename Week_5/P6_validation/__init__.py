"""
P6: Validation Module
Bu modül time series model validation ve cross-validation işlemlerini içerir.

Modüller:
- time_series_cv.py: Comprehensive cross-validation (Class-based)
- time_series_cv_simple.py: Simplified cross-validation (Function-based)

Usage:
    # Comprehensive (Class-based)
    from P6_validation.time_series_cv import TimeSeriesCV
    tscv = TimeSeriesCV(validation_horizon=28, n_splits=3)
    cv_results, summary = tscv.run_full_pipeline()
    
    # Simple (Function-based)
    from P6_validation.time_series_cv_simple import run_time_series_cv
    cv_results, summary = run_time_series_cv()
"""

__version__ = "1.0.0"
__author__ = "M5 Forecasting Team"

try:
    from .time_series_cv import TimeSeriesCV
    from .time_series_cv_simple import run_time_series_cv
    
    __all__ = ['TimeSeriesCV', 'run_time_series_cv']
    
except ImportError as e:
    print(f"⚠️  P6_validation import warning: {e}")
    __all__ = []