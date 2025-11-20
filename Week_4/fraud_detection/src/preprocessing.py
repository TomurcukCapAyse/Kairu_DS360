"""
Feature Scaling ve Encoding utilities
Fraud detection için veri ön işleme araçları
"""
"""
Cleaned and fixed preprocessing utilities for fraud_detection.
This file is a drop-in replacement candidate for `preprocessing.py`.
Key fixes:
- Per-column LabelEncoder handling
- Robust OneHotEncoder fallback for sklearn versions
- Safe transform with unknown labels mapped to -1
- visualize_distributions saving uses pathlib.Path and creates parent dir
- No non-breaking spaces; normalized whitespace
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """Feature preprocessing for fraud detection."""

    def __init__(self, scaling_method='robust', encoding_method='onehot'):
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method

        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaling_method must be 'standard', 'robust' or 'minmax'")

        if encoding_method == 'onehot':
            try:
                self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            except TypeError:
                self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        elif encoding_method == 'label':
            self.label_encoders = {}
        elif encoding_method == 'ordinal':
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        else:
            raise ValueError("encoding_method must be 'onehot', 'label' or 'ordinal'")

        self.numerical_features = []
        self.categorical_features = []
        self.encoded_feature_names = []
        self.is_fitted = False
        self.numerical_features_to_scale = []

    def identify_features(self, df):
        self.numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    def fit_transform(self, df, target_col=None):
        df = df.copy()
        if target_col and target_col in df.columns:
            target = df[target_col].copy()
            df = df.drop(columns=[target_col])
        else:
            target = None

        self.identify_features(df)

        if self.numerical_features:
            df[self.numerical_features] = self.scaler.fit_transform(df[self.numerical_features])
            self.numerical_features_to_scale = self.numerical_features

        if self.categorical_features:
            if self.encoding_method == 'onehot':
                encoded = self.encoder.fit_transform(df[self.categorical_features])
                try:
                    self.encoded_feature_names = self.encoder.get_feature_names_out(self.categorical_features).tolist()
                except Exception:
                    self.encoded_feature_names = [f"c_{i}" for i in range(encoded.shape[1])]
                encoded_df = pd.DataFrame(encoded, columns=self.encoded_feature_names, index=df.index)
                df = df.drop(columns=self.categorical_features)
                df = pd.concat([df, encoded_df], axis=1)

            elif self.encoding_method == 'label':
                for col in self.categorical_features:
                    le = LabelEncoder()
                    df[col] = df[col].astype(str)
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le

            elif self.encoding_method == 'ordinal':
                df[self.categorical_features] = self.encoder.fit_transform(df[self.categorical_features])

        self.is_fitted = True
        if target is not None:
            df[target_col] = target
        return df

    def transform(self, df, target_col=None):
        if not self.is_fitted:
            raise ValueError("Call fit_transform first")
        df = df.copy()
        if target_col and target_col in df.columns:
            target = df[target_col].copy()
            df = df.drop(columns=[target_col])
        else:
            target = None

        if self.numerical_features_to_scale:
            df[self.numerical_features_to_scale] = self.scaler.transform(df[self.numerical_features_to_scale])

        if self.categorical_features:
            if self.encoding_method == 'onehot':
                encoded = self.encoder.transform(df[self.categorical_features])
                encoded_df = pd.DataFrame(encoded, columns=self.encoded_feature_names, index=df.index)
                df = df.drop(columns=self.categorical_features)
                df = pd.concat([df, encoded_df], axis=1)

            elif self.encoding_method == 'label':
                for col in self.categorical_features:
                    le = self.label_encoders.get(col)
                    if le is None:
                        raise ValueError(f"Missing LabelEncoder for {col}")
                    vals = df[col].astype(str).values
                    try:
                        transformed = le.transform(vals)
                    except ValueError:
                        known = set(le.classes_)
                        transformed = np.array([le.transform([v])[0] if v in known else -1 for v in vals])
                    df[col] = transformed

            elif self.encoding_method == 'ordinal':
                df[self.categorical_features] = self.encoder.transform(df[self.categorical_features])

        if target is not None:
            df[target_col] = target
        return df

    def visualize_distributions(self, before, after, n_features=6, save_path=None):
        common = list(set(before.columns) & set(after.columns))
        cols = common[:min(n_features, len(common))]
        if not cols:
            logger.warning("No shared columns for visualization")
            return
        fig, axes = plt.subplots(len(cols), 2, figsize=(12, 4 * len(cols)))
        if len(cols) == 1:
            axes = axes.reshape(1, -1)
        for i, c in enumerate(cols):
            axes[i, 0].hist(before[c].dropna(), bins=50, color='steelblue', edgecolor='black')
            axes[i, 0].set_title(f"{c} - before")
            axes[i, 1].hist(after[c].dropna(), bins=50, color='orange', edgecolor='black')
            axes[i, 1].set_title(f"{c} - after")
        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=150)
            logger.info(f"Saved visualization to {save_path}")
        plt.close()


class ImbalanceHandler:
    @staticmethod
    def analyze_imbalance(y, class_names=None):
        from collections import Counter
        counts = Counter(y)
        total = len(y)
        for cls, cnt in counts.items():
            label = class_names[cls] if class_names and cls in range(len(class_names)) else f"Class {cls}"
            print(f"{label}: {cnt} ({cnt/total*100:.2f}%)")
        if len(counts) == 2:
            minority = min(counts.values())
            majority = max(counts.values())
            print(f"Imbalance ratio: {majority/minority:.2f}:1")

    @staticmethod
    def apply_smote(X, y, sampling_strategy='auto'):
        sm = SMOTE(sampling_strategy=sampling_strategy)
        return sm.fit_resample(X, y)

    @staticmethod
    def apply_adasyn(X, y, sampling_strategy='auto'):
        ad = ADASYN(sampling_strategy=sampling_strategy)
        return ad.fit_resample(X, y)

    @staticmethod
    def apply_undersample(X, y, sampling_strategy='auto'):
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
        return rus.fit_resample(X, y)

    @staticmethod
    def apply_smote_tomek(X, y, sampling_strategy='auto'):
        smt = SMOTETomek(sampling_strategy=sampling_strategy)
        return smt.fit_resample(X, y)


def demo_preprocessing(save_outputs: bool = True, out_dir: Path | None = None, save_visuals: bool = False, verbose: bool = False):
    """Run a small preprocessing demo.

    Args:
        save_outputs (bool): If True, saves `train_processed_supervised.csv`,
            `test_processed_supervised.csv` and `dataset_processed_supervised.csv` to `out_dir`.
        out_dir (Path|None): Target directory for outputs. Defaults to
            `fraud_detection/data/processed` (module parent `data/processed`).

    Returns:
        tuple: (train_processed_df, test_processed_df, preprocessor_instance)
    """
    BASE_DIR = Path(__file__).resolve().parent.parent
    if out_dir is None:
        out_dir = BASE_DIR / 'data' / 'processed'
    out_dir = Path(out_dir)

    in_path = out_dir / 'dataset_with_anomaly_scores_raw.csv'
    if not in_path.exists():
        raise FileNotFoundError(f"Expected input file not found: {in_path}")

    df = pd.read_csv(in_path)
    assert 'Class' in df.columns
    if 'split' in df.columns:
        feature_cols = [c for c in df.columns if c not in ('Class', 'split')]
        train = df[df['split'] == 'train'].reset_index(drop=True)
        test = df[df['split'] == 'test'].reset_index(drop=True)
    else:
        X = df.drop(columns=['Class'])
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    pre = FeaturePreprocessor(scaling_method='robust', encoding_method='onehot')
    train_proc = pre.fit_transform(train.copy(), target_col='Class')
    test_proc = pre.transform(test.copy(), target_col='Class')

    if save_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
        train_out = out_dir / 'train_processed_supervised.csv'
        test_out = out_dir / 'test_processed_supervised.csv'
        full_out = out_dir / 'dataset_processed_supervised.csv'

        pd.concat([train_proc.reset_index(drop=True)], axis=0)
        train_proc.to_csv(train_out, index=False)
        test_proc.to_csv(test_out, index=False)

        # Save combined full dataset (if split exists preserve split column)
        if 'split' in df.columns:
            full_df = pd.concat([
                train_proc.reset_index(drop=True),
                test_proc.reset_index(drop=True)
            ], axis=0, ignore_index=True)
        else:
            full_df = pd.concat([
                train_proc.reset_index(drop=True),
                test_proc.reset_index(drop=True)
            ], axis=0, ignore_index=True)
        full_df.to_csv(full_out, index=False)

        if verbose:
            print(f"Saved outputs to: {train_out}, {test_out}, {full_out}")

    # Optionally save a visualization comparing feature distributions
    if save_visuals:
        # Use the original train (before) and the processed train (after)
        before_cols = [c for c in train.columns if c not in ('Class', 'split')]
        after_cols = [c for c in train_proc.columns if c not in ('Class', 'split')]
        # select common/shared feature names where possible; for one-hot, we'll pass overlapping columns
        before_df = train[before_cols].copy()
        # try to align after_df to have comparable numeric columns where names match
        after_df = train_proc.copy()
        plot_path = out_dir / 'preprocessing_comparison.png'
        try:
            pre.visualize_distributions(before_df, after_df, n_features=6, save_path=plot_path)
            if verbose:
                print(f"Saved preprocessing comparison plot to: {plot_path}")
        except Exception as e:
            logger.warning(f"Could not save visualization: {e}")

    return train_proc, test_proc, pre


if __name__ == '__main__':
    demo_preprocessing()