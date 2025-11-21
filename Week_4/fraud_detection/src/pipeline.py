"""
Ana Fraud Detection Pipeline
Training, inference ve deployment için end-to-end pipeline
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
import logging
import argparse
from datetime import datetime
import warnings

# Local imports
try:
    from preprocessing import FeaturePreprocessor, ImbalanceHandler
except Exception:
    FeaturePreprocessor = None
    ImbalanceHandler = None

try:
    from evaluation import FraudEvaluator
except Exception:
    class FraudEvaluator:
        def __init__(self):
            self.results = {}
        def evaluate(self, y_true, y_pred, y_prob=None):
            # minimal placeholder
            self.results = {"roc_auc": 0.0}

try:
    from explainability_clean import ModelExplainer
except Exception:
    ModelExplainer = None

try:
    from outlier_detection import OutlierDetector
except Exception:
    OutlierDetector = None

warnings.filterwarnings('ignore')


class FraudDetectionPipeline:
    """End-to-end Fraud Detection Pipeline"""
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Args:
            config_path (str): Configuration dosyası yolu
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_mlflow()
        
        # Pipeline components
        self.preprocessor = None
        self.models = {}
        self.evaluators = {}
        self.explainer = None
        
        # Data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.logger.info("Fraud Detection Pipeline initialized")
    
    def _load_config(self, config_path):
        """Configuration dosyasını yükle"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            # Default config if file not found
            return self._get_default_config()
    
    def _get_default_config(self):
        """Default configuration"""
        # Minimal default configuration
        return {
            'random_state': 42,
            'test_size': 0.2,
            'models': ['random_forest'],
            'max_train_samples': 100000,
            'n_estimators': 50,
            'n_jobs': 1,
            'save_models': True,
            'scaling_method': 'robust',
            'encoding_method': 'onehot',
        }

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger(__name__)

    def _setup_mlflow(self):
        # Minimal MLflow setup placeholder
        try:
            mlflow.set_tracking_uri(self.config.get('mlflow', {}).get('tracking_uri', 'file://./mlruns'))
        except Exception:
            pass
    
    def run_full_pipeline(self, data_path=None, save_models=True, use_kagglehub=False):
        """Full pipeline execution"""
        self.logger.info("Full Fraud Detection Pipeline başlatılıyor...")
        
        try:
            # 1. Load data
            self.load_data(data_path, synthetic=(data_path is None and not use_kagglehub),
                          download_with_kagglehub=use_kagglehub)
            
            # 2. Preprocess
            self.preprocess_data()
            
            # 3. Train models
            self.train_models()
            
            # 4. Evaluate models
            self.evaluate_models()
            
            # 5. Explain best model
            best_model = self._find_best_model()
            self.explain_models(best_model)
            
            # 6. Save models
            if save_models:
                self.save_models()
            
            self.logger.info("Full pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return False

    # --- Minimal implementations / stubs to allow demo runs ---
    def load_data(self, data_path=None, synthetic=False, download_with_kagglehub=False):
        """Load data. If synthetic True, load a small local processed CSV as demo."""
        # Prefer processed dataset if available. If synthetic requested, use processed files under data/processed
        processed_dir = Path(__file__).resolve().parent.parent / 'data' / 'processed'
        train_path = processed_dir / 'train_processed_supervised.csv'
        test_path = processed_dir / 'test_processed_supervised.csv'
        combined_path = processed_dir / 'dataset_processed_supervised.csv'

        if synthetic:
            # If processed split files exist, load them
            if train_path.exists() and test_path.exists():
                train_proc = pd.read_csv(train_path)
                test_proc = pd.read_csv(test_path)
            elif combined_path.exists():
                df = pd.read_csv(combined_path)
                if 'split' in df.columns:
                    train_proc = df[df['split'] == 'train'].drop(columns=['split']).reset_index(drop=True)
                    test_proc = df[df['split'] == 'test'].drop(columns=['split']).reset_index(drop=True)
                else:
                    # simple stratified split
                    from sklearn.model_selection import train_test_split as _tts
                    X = df.drop(columns=['Class'])
                    y = df['Class']
                    X_train, X_test, y_train, y_test = _tts(X, y, test_size=0.2, random_state=int(self.config.get('random_state', 42)), stratify=y if y.nunique()>1 else None)
                    train_proc = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
                    test_proc = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
            else:
                # Fall back to preprocessing demo utility if available
                try:
                    from preprocessing import demo_preprocessing
                    train_proc, test_proc, pre = demo_preprocessing(save_outputs=True, verbose=False)
                except Exception:
                    self.logger.warning('Demo processed data not found; creating tiny synthetic dataset')
                    df = pd.DataFrame({f'v{i}': np.random.randn(100) for i in range(1, 6)})
                    df['Class'] = np.random.choice([0, 1], size=len(df), p=[0.98, 0.02])
                    X = df.drop(columns=['Class'])
                    y = df['Class']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(self.config.get('random_state', 42)), stratify=y if y.nunique()>1 else None)
                    train_proc = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
                    test_proc = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
        else:
            if data_path is None:
                raise ValueError('data_path required when synthetic=False')
            df = pd.read_csv(data_path)
            if 'Class' not in df.columns:
                df['Class'] = 0
            X = df.drop(columns=['Class'])
            y = df['Class']
            rs = int(self.config.get('random_state', 42))
            test_size = float(self.config.get('test_size', 0.2))
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=rs, stratify=y if y.nunique()>1 else None)
            self.X_test_processed = self.X_test.copy()

        # If we loaded processed train/test, assign to pipeline attributes
        if 'train_proc' in locals() and 'test_proc' in locals():
            if 'Class' not in train_proc.columns:
                raise ValueError('Processed train data does not contain Class column')
            X_tr = train_proc.drop(columns=['Class'])
            y_tr = train_proc['Class']
            X_te = test_proc.drop(columns=['Class'])
            y_te = test_proc['Class']
            self.X_train, self.X_test, self.y_train, self.y_test = X_tr, X_te, y_tr, y_te
            self.X_test_processed = self.X_test.copy()

    def preprocess_data(self):
        # Integrate FeaturePreprocessor if available
        if FeaturePreprocessor is None:
            self.logger.info('Preprocess: FeaturePreprocessor not available, skipping (no-op)')
            return

        self.logger.info('Preprocess: running FeaturePreprocessor')
        # Recombine X and y into DataFrame expected by preprocessor
        try:
            train_df = pd.concat([self.X_train.reset_index(drop=True), self.y_train.reset_index(drop=True)], axis=1)
        except Exception:
            train_df = pd.concat([pd.DataFrame(self.X_train).reset_index(drop=True), pd.Series(self.y_train).reset_index(drop=True)], axis=1)

        # target column name expected 'Class'
        if 'Class' not in train_df.columns:
            train_df.columns = list(train_df.columns[:-1]) + ['Class']

        pre = FeaturePreprocessor(scaling_method=self.config.get('scaling_method', 'robust'),
                                  encoding_method=self.config.get('encoding_method', 'onehot'))
        train_proc = pre.fit_transform(train_df.copy(), target_col='Class')

        # Process test
        try:
            test_df = pd.concat([self.X_test.reset_index(drop=True), self.y_test.reset_index(drop=True)], axis=1)
        except Exception:
            test_df = pd.concat([pd.DataFrame(self.X_test).reset_index(drop=True), pd.Series(self.y_test).reset_index(drop=True)], axis=1)
        if 'Class' not in test_df.columns:
            test_df.columns = list(test_df.columns[:-1]) + ['Class']

        test_proc = pre.transform(test_df.copy(), target_col='Class')

        # Assign back
        self.preprocessor = pre
        self.X_train = train_proc.drop(columns=['Class'])
        self.y_train = train_proc['Class']
        self.X_test = test_proc.drop(columns=['Class'])
        self.y_test = test_proc['Class']
        self.X_test_processed = self.X_test.copy()

    def train_models(self):
        # Train a RandomForest with optional stratified subsampling for large datasets
        self.logger.info('Training RandomForest (with subsample if needed)')
        rs = int(self.config.get('random_state', 42))
        max_train = int(self.config.get('max_train_samples', 50000))
        n_estimators = int(self.config.get('n_estimators', 50))
        n_jobs = int(self.config.get('n_jobs', 1))

        X_train = self.X_train
        y_train = self.y_train

        # If training set is large, perform stratified subsample to keep training reasonable
        if len(X_train) > max_train:
            self.logger.info(f'Training set size {len(X_train)} > {max_train}, performing stratified subsample')
            try:
                from sklearn.model_selection import StratifiedShuffleSplit
                sss = StratifiedShuffleSplit(n_splits=1, train_size=max_train, random_state=rs)
                # sss.split expects array-like
                X_for_split = X_train.values if hasattr(X_train, 'values') else X_train
                y_for_split = y_train.values if hasattr(y_train, 'values') else y_train
                train_idx, _ = next(sss.split(X_for_split, y_for_split))
                if hasattr(X_train, 'iloc'):
                    X_sub = X_train.iloc[train_idx]
                else:
                    X_sub = X_train[train_idx]
                if hasattr(y_train, 'iloc'):
                    y_sub = y_train.iloc[train_idx]
                else:
                    y_sub = y_train[train_idx]
            except Exception as e:
                self.logger.warning(f'Stratified subsample failed ({e}), falling back to random sample')
                rng = np.random.RandomState(rs)
                idx = rng.choice(np.arange(len(X_train)), size=max_train, replace=False)
                X_sub = X_train.iloc[idx] if hasattr(X_train, 'iloc') else X_train[idx]
                y_sub = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
        else:
            X_sub, y_sub = X_train, y_train

        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=rs, n_jobs=n_jobs)
        rf.fit(X_sub, y_sub)
        self.models['random_forest'] = rf

    def evaluate_models(self):
        self.logger.info('Evaluating models (minimal)')
        for name, model in self.models.items():
            probs = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(len(self.X_test))
            preds = model.predict(self.X_test)
            ev = FraudEvaluator(model=model, model_name=name)
            try:
                # Use the evaluation module's binary evaluation when available
                if hasattr(ev, 'evaluate_binary_classification'):
                    results = ev.evaluate_binary_classification(self.X_test, self.y_test, y_pred_proba=probs)
                    ev.results = results
                else:
                    # fallback to generic evaluator interface
                    ev.evaluate(self.y_test, preds, probs)
            except Exception as e:
                self.logger.warning(f'Evaluation failed for {name}: {e}')
                ev.results = {'roc_auc': 0.0}
            self.evaluators[name] = ev

    def explain_models(self, model_name):
        self.logger.info(f'Explain model: {model_name} (stub)')
        # Minimal explanation: return feature importances if available
        model = self.models.get(model_name)
        if model is None:
            return {}, {}
        if hasattr(model, 'feature_importances_'):
            fi = {f'feat_{i}': v for i, v in enumerate(model.feature_importances_)}
            return fi, {}
        return {}, {}

    def save_models(self):
        self.logger.info('Saving models (joblib)')
        out_dir = Path('models')
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, out_dir / f'{name}.joblib')

    def load_models(self):
        self.logger.info('Loading models from models/ (if present)')
        out_dir = Path('models')
        for p in out_dir.glob('*.joblib'):
            name = p.stem
            try:
                self.models[name] = joblib.load(p)
            except Exception:
                self.logger.warning(f'Could not load {p}')

    def predict(self, data_processed, model_name='random_forest'):
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f'Model {model_name} not found')
        preds = model.predict(data_processed)
        probs = model.predict_proba(data_processed)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(len(data_processed))
        return preds, probs
    
    def _find_best_model(self):
        """En iyi modeli bul (ROC-AUC'ye göre)"""
        best_model = None
        best_score = 0
        
        for model_name, evaluator in self.evaluators.items():
            if evaluator.results and 'roc_auc' in evaluator.results:
                roc_auc = evaluator.results['roc_auc']
                if roc_auc > best_score:
                    best_score = roc_auc
                    best_model = model_name
        
        self.logger.info(f"Best model: {best_model} (ROC-AUC: {best_score:.4f})")
        return best_model or 'random_forest'


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(description='Fraud Detection Pipeline')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--data', help='Data file path (optional, uses synthetic if not provided)')
    parser.add_argument('--mode', choices=['train', 'predict', 'explain'], default='train', help='Pipeline mode')
    parser.add_argument('--model', default='random_forest', help='Model name for prediction/explanation')
    parser.add_argument('--load_models', action='store_true', help='Load existing models')
    parser.add_argument('--save_models', action='store_true', help='Save trained models')
    parser.add_argument('--use_kagglehub', action='store_true', help='Download data with KaggleHub')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FraudDetectionPipeline(args.config)
    
    if args.mode == 'train':
        # Training mode
        # Determine whether to save models: prefer CLI flag, otherwise use config default
        save_models_flag = args.save_models or bool(pipeline.config.get('save_models', True))
        success = pipeline.run_full_pipeline(args.data, save_models_flag, args.use_kagglehub)
        sys.exit(0 if success else 1)
        
    elif args.mode == 'predict':
        # Prediction mode
        if args.load_models:
            pipeline.load_models()
        
        # Demo prediction with synthetic data
        pipeline.load_data(synthetic=True)
        pipeline.preprocess_data()

        # If model not loaded, train a quick demo model
        if args.model not in pipeline.models:
            pipeline.train_models()

        predictions, probabilities = pipeline.predict(
            pipeline.X_test_processed.head(), args.model
        )
        
        print("Sample Predictions:")
        for i, (pred, prob) in enumerate(zip(predictions[:5], probabilities[:5])):
            print(f"Sample {i}: Prediction={pred}, Probability={prob:.4f}")
    
    elif args.mode == 'explain':
        # Explanation mode
        if args.load_models:
            pipeline.load_models()
        
        # Load data and explain
        pipeline.load_data(synthetic=True)
        pipeline.preprocess_data()
        
        importance, patterns = pipeline.explain_models(args.model)
        
        print("Top 10 Important Features:")
        for i, (feature, score) in enumerate(list(importance.items())[:10]):
            print(f"{i+1}. {feature}: {score:.4f}")


if __name__ == "__main__":
    main()