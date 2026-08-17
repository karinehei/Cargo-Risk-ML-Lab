"""Model training, prediction and utilities."""

from src.models.compare import ComparisonResult, compare_models
from src.models.estimators import (
    NEEDS_SCALING,
    XGBTrainWeightedClassifier,
    build_estimator,
    build_model_pipeline,
)
from src.models.inference import prepare_inference_frame
from src.models.train import (
    TrainedModelBundle,
    load_model_bundle,
    predict_labels,
    predict_proba,
    save_model_bundle,
    train_model,
)

__all__ = [
    "NEEDS_SCALING",
    "ComparisonResult",
    "TrainedModelBundle",
    "XGBTrainWeightedClassifier",
    "build_estimator",
    "build_model_pipeline",
    "compare_models",
    "load_model_bundle",
    "predict_labels",
    "predict_proba",
    "prepare_inference_frame",
    "save_model_bundle",
    "train_model",
]
