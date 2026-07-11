"""Safe, optional loading and inference for the AMD-trained FRA artifact."""

from __future__ import annotations

import importlib
import json
import logging
import pickle
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.models.fra_cnn import build_fra_1d_cnn
from app.schemas.diagnosis import DiagnosticAnalysisResult


logger = logging.getLogger(__name__)

FRA_SEQUENCE_LENGTH = 256
FRA_REQUIRED_COLUMNS = ("frequency_hz", "magnitude_db", "phase_deg")
FRA_LABELS = (
    "healthy",
    "winding_deformation_suspected",
    "core_clamping_issue_suspected",
    "insulation_related_abnormality_suspected",
    "needs_human_review",
)

FRAArtifactPredictor = Callable[[pd.DataFrame], DiagnosticAnalysisResult]
ModelFactory = Callable[[Any, int, int], Any]

_AUTO_TORCH = object()


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = float(values.mean())
    standard_deviation = float(values.std())
    if standard_deviation <= np.finfo(np.float64).eps:
        return np.zeros_like(values)
    return (values - mean) / standard_deviation


def preprocess_fra_frame(
    frame: pd.DataFrame,
    *,
    sequence_length: int = FRA_SEQUENCE_LENGTH,
) -> np.ndarray:
    """Resample an FRA curve into log-frequency, magnitude, and phase channels."""

    missing = [column for column in FRA_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required FRA columns: {', '.join(missing)}")
    if len(frame) < 2:
        raise ValueError("FRA model input requires at least 2 rows")
    if sequence_length < 4:
        raise ValueError("FRA model sequence length must be at least 4")

    frequency = frame["frequency_hz"].to_numpy(dtype=float)
    magnitude = frame["magnitude_db"].to_numpy(dtype=float)
    phase = frame["phase_deg"].to_numpy(dtype=float)
    if not (
        np.isfinite(frequency).all() and np.isfinite(magnitude).all() and np.isfinite(phase).all()
    ):
        raise ValueError("FRA model input contains non-finite values")
    if (frequency <= 0).any():
        raise ValueError("FRA model frequencies must be greater than zero")

    order = np.argsort(frequency)
    log_frequency = np.log10(frequency[order])
    if (np.diff(log_frequency) <= 0).any():
        raise ValueError("FRA model frequencies must be unique")

    grid = np.linspace(float(log_frequency[0]), float(log_frequency[-1]), sequence_length)
    resampled_magnitude = np.interp(grid, log_frequency, magnitude[order])
    unwrapped_phase = np.rad2deg(np.unwrap(np.deg2rad(phase[order])))
    resampled_phase = np.interp(grid, log_frequency, unwrapped_phase)

    channels = np.stack(
        (
            _standardize(grid),
            _standardize(resampled_magnitude),
            _standardize(resampled_phase),
        )
    )
    return np.ascontiguousarray(channels, dtype=np.float32)


def _read_label_map(path: Path) -> tuple[str, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("FRA label map must be a JSON object")
    expected_keys = {str(index) for index in range(len(FRA_LABELS))}
    if set(raw) != expected_keys:
        raise ValueError("FRA label map must contain exactly the class keys 0 through 4")
    labels = tuple(raw[str(index)] for index in range(len(FRA_LABELS)))
    if labels != FRA_LABELS:
        raise ValueError("FRA label map does not match the five-class training contract")
    return labels


def _resolve_torch(torch_module: Any) -> Any | None:
    if torch_module is not _AUTO_TORCH:
        return torch_module
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        logger.warning("FRA artifact disabled because PyTorch is unavailable: %s", exc)
        return None


def create_fra_artifact_predictor(
    model_path: Path,
    label_map_path: Path,
    *,
    torch_module: Any = _AUTO_TORCH,
    model_factory: ModelFactory = build_fra_1d_cnn,
) -> FRAArtifactPredictor | None:
    """Load the optional CPU inference artifact or return ``None`` for rule fallback."""

    model_path = Path(model_path)
    label_map_path = Path(label_map_path)
    if not model_path.is_file() or not label_map_path.is_file():
        logger.warning(
            "FRA model fallback active; artifact or label map is missing (%s, %s)",
            model_path,
            label_map_path,
        )
        return None

    try:
        labels = _read_label_map(label_map_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("FRA model fallback active; label map is invalid: %s", exc)
        return None

    torch = _resolve_torch(torch_module)
    if torch is None:
        return None

    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError("FRA artifact must contain a non-empty state_dict")
        model = model_factory(torch, len(FRA_REQUIRED_COLUMNS), len(labels))
        model.load_state_dict(state_dict, strict=True)
        model.to("cpu")
        model.eval()
    except (EOFError, OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        logger.warning("FRA model fallback active; artifact could not be loaded: %s", exc)
        return None

    def predict(frame: pd.DataFrame) -> DiagnosticAnalysisResult:
        features = preprocess_fra_frame(frame)
        inputs = torch.from_numpy(features).unsqueeze(0)
        with torch.inference_mode():
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(labels),) or not np.isfinite(probabilities).all():
            raise RuntimeError("FRA model returned invalid class probabilities")

        predicted_index = int(np.argmax(probabilities))
        fault_class = labels[predicted_index]
        confidence = float(probabilities[predicted_index])
        healthy_probability = float(probabilities[labels.index("healthy")])
        anomaly_score = float(np.clip(1.0 - healthy_probability, 0.0, 1.0))
        is_anomalous = fault_class != "healthy"
        class_probabilities = {
            f"probability_{label}": round(float(probability), 6)
            for label, probability in zip(labels, probabilities, strict=True)
        }

        return DiagnosticAnalysisResult(
            fault_class=fault_class,
            is_anomalous=is_anomalous,
            confidence=round(confidence, 6),
            anomaly_score=round(anomaly_score, 6),
            evidence=[
                "AMD-trained FRA artifact classified the resampled three-channel curve as "
                f"{fault_class} with {confidence:.1%} confidence.",
                "Model output is decision support only and requires qualified human review.",
            ],
            metrics=class_probabilities,
            analysis_method="fra_model_artifact",
            requires_human_review=True,
            anomaly_types=[] if not is_anomalous else ["fra_model_classification"],
            likely_faults=[] if not is_anomalous else [fault_class],
        )

    return predict
