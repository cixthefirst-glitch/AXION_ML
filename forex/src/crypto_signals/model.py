from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.crypto_signals.features import FEATURE_COLUMNS, get_feature_columns, prepare_training_data
from src.crypto_signals.persistence import ModelMetadata, load_model, save_model


class SignalModel:
    def __init__(self, model_path: str, model_type: str = "xgboost") -> None:
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.pipeline: Optional[Pipeline] = None
        self.encoder = LabelEncoder()
        self.feature_columns = get_feature_columns()
        self.metadata: Optional[Dict[str, Any]] = None

    def build_pipeline(self) -> Pipeline:
        if self.model_type == "xgboost":
            estimator = XGBClassifier(
                objective="multi:softprob",
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )

        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    def fit(self, features: pd.DataFrame, labels: pd.Series, test_size: float = 0.15) -> Dict[str, float]:
        X = features[self.feature_columns].copy()
        label_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        y = labels.map(label_map).astype(str).copy()
        y_encoded = self.encoder.fit_transform(y)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y_encoded,
            test_size=test_size,
            shuffle=False,
        )

        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(
            X_train,
            y_train,
            model__sample_weight=sample_weights,
            model__eval_set=[(X_val, y_val)],
            model__early_stopping_rounds=20,
            model__verbose=False,
        )

        y_pred = self.pipeline.predict(X_val)
        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
        }
        return metrics

    def predict(self, features: pd.DataFrame) -> List[str]:
        if self.pipeline is None:
            raise ValueError("Model pipeline is not loaded")
        labels = self.pipeline.predict(features[self.feature_columns])
        return self.encoder.inverse_transform(labels.tolist()).tolist()

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("Model pipeline is not loaded")
        return self.pipeline.predict_proba(features[self.feature_columns])

    def save(self) -> None:
        if self.pipeline is None:
            raise ValueError("No trained model to save")
        self.metadata = {
            "version": "v1",
            "created_at": datetime.utcnow().isoformat(),
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
        }
        payload = {
            "pipeline": self.pipeline,
            "encoder": self.encoder,
            "metadata": self.metadata,
        }
        save_model(payload, self.model_path, ModelMetadata(version="v1", created_at=self.metadata["created_at"], config={"model_type": self.model_type}))

    def load(self) -> bool:
        payload = load_model(self.model_path)
        if payload is None:
            return False
        stored = payload["model"]
        if isinstance(stored, dict):
            self.pipeline = stored.get("pipeline")
            self.encoder = stored.get("encoder", self.encoder)
            self.metadata = stored.get("metadata")
        else:
            self.pipeline = stored
            self.metadata = payload.get("metadata")
        return self.pipeline is not None

    def exists(self) -> bool:
        from pathlib import Path

        return Path(self.model_path).exists()

    @classmethod
    def build_training_dataset(cls, frames: List[pd.DataFrame], horizon: int = 6, reward_risk: float = 2.0) -> pd.DataFrame:
        label_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        rows = []
        for frame in frames:
            features, labels = prepare_training_data(frame, horizon=horizon, reward_risk=reward_risk)
            combined = features.copy()
            combined["target"] = labels.map(label_map)
            rows.append(combined)
        if not rows:
            raise ValueError("No training frames provided")
        return pd.concat(rows, axis=0, ignore_index=True)
