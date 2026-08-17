"""Model evaluation metrics, plots and error analysis."""

from src.evaluation.bootstrap import bootstrap_metric_intervals, intervals_to_records
from src.evaluation.errors import build_error_analysis_report, label_errors
from src.evaluation.latency import measure_inference_latency
from src.evaluation.metrics import (
    analyse_errors,
    compute_classification_metrics,
    confusion_and_report,
    evaluate_predictions,
    expected_calibration_error,
    save_comparison_plot,
    save_evaluation_plots,
    save_threshold_plot,
)
from src.evaluation.operations import compare_operating_points, operational_rates_per_1000
from src.evaluation.threshold import select_threshold

__all__ = [
    "analyse_errors",
    "bootstrap_metric_intervals",
    "build_error_analysis_report",
    "compare_operating_points",
    "compute_classification_metrics",
    "confusion_and_report",
    "evaluate_predictions",
    "expected_calibration_error",
    "intervals_to_records",
    "label_errors",
    "measure_inference_latency",
    "operational_rates_per_1000",
    "save_comparison_plot",
    "save_evaluation_plots",
    "save_threshold_plot",
    "select_threshold",
]
