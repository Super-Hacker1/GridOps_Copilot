"""Tests for the optional AMD-trained FRA model artifact integration."""

from __future__ import annotations

import json
import pickle
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.models.fra_cnn import build_fra_1d_cnn
from app.models.fra_model_loader import (
    create_fra_artifact_predictor,
    preprocess_fra_frame,
)


LABEL_MAP = {
    "0": "healthy",
    "1": "winding_deformation_suspected",
    "2": "core_clamping_issue_suspected",
    "3": "insulation_related_abnormality_suspected",
    "4": "needs_human_review",
}


def make_fra_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frequency_hz": [100.0, 1.0, 10.0],
            "magnitude_db": [20.0, 0.0, 10.0],
            "phase_deg": [90.0, -90.0, 0.0],
        }
    )


def test_preprocess_fra_frame_returns_repeatable_three_channel_sequence() -> None:
    first = preprocess_fra_frame(make_fra_frame(), sequence_length=5)
    second = preprocess_fra_frame(make_fra_frame(), sequence_length=5)

    assert first.shape == (3, 5)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.mean(axis=1), np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(first.std(axis=1), np.ones(3), atol=1e-6)
    assert np.all(np.diff(first, axis=1) > 0)


def test_preprocess_fra_frame_unwraps_phase_before_resampling() -> None:
    baseline_frequency = np.logspace(1.0, 6.0, 128)
    current_frequency = np.logspace(1.01, 5.99, 127)

    def frame(frequency: np.ndarray) -> pd.DataFrame:
        unwrapped_phase = 160.0 + 40.0 * ((np.log10(frequency) - 1.0) / 5.0)
        return pd.DataFrame(
            {
                "frequency_hz": frequency,
                "magnitude_db": np.linspace(0.0, -10.0, len(frequency)),
                "phase_deg": (unwrapped_phase + 180.0) % 360.0 - 180.0,
            }
        )

    baseline = preprocess_fra_frame(frame(baseline_frequency))
    current = preprocess_fra_frame(frame(current_frequency))

    np.testing.assert_allclose(current[2], baseline[2], atol=1e-4)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (
            pd.DataFrame({"frequency_hz": [1.0, 2.0], "magnitude_db": [0.0, 1.0]}),
            "phase_deg",
        ),
        (
            pd.DataFrame(
                {
                    "frequency_hz": [1.0, 1.0],
                    "magnitude_db": [0.0, 1.0],
                    "phase_deg": [0.0, 1.0],
                }
            ),
            "unique",
        ),
        (
            pd.DataFrame(
                {
                    "frequency_hz": [1.0, 2.0],
                    "magnitude_db": [0.0, np.nan],
                    "phase_deg": [0.0, 1.0],
                }
            ),
            "non-finite",
        ),
    ],
)
def test_preprocess_fra_frame_rejects_invalid_curves(
    frame: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        preprocess_fra_frame(frame)


def test_artifact_factory_returns_none_when_files_are_missing(tmp_path: Path) -> None:
    assert (
        create_fra_artifact_predictor(
            tmp_path / "missing.pt",
            tmp_path / "missing-labels.json",
            torch_module=None,
        )
        is None
    )


def test_artifact_factory_returns_none_for_invalid_label_map(tmp_path: Path) -> None:
    model_path = tmp_path / "fra.pt"
    labels_path = tmp_path / "labels.json"
    model_path.write_bytes(b"artifact")
    labels_path.write_text(json.dumps({"0": "healthy"}), encoding="utf-8")
    fake_torch = SimpleNamespace(load=lambda *args, **kwargs: pytest.fail("must not load model"))

    assert (
        create_fra_artifact_predictor(
            model_path,
            labels_path,
            torch_module=fake_torch,
        )
        is None
    )


def test_artifact_factory_returns_none_when_torch_is_unavailable(tmp_path: Path) -> None:
    model_path = tmp_path / "fra.pt"
    labels_path = tmp_path / "labels.json"
    model_path.write_bytes(b"artifact")
    labels_path.write_text(json.dumps(LABEL_MAP), encoding="utf-8")

    assert (
        create_fra_artifact_predictor(
            model_path,
            labels_path,
            torch_module=None,
        )
        is None
    )


@pytest.mark.parametrize(
    "load_error",
    [
        RuntimeError("invalid"),
        pickle.UnpicklingError("unsafe"),
        EOFError("truncated"),
    ],
)
def test_artifact_factory_returns_none_when_state_dict_loading_fails(
    tmp_path: Path,
    load_error: Exception,
) -> None:
    model_path = tmp_path / "fra.pt"
    labels_path = tmp_path / "labels.json"
    model_path.write_bytes(b"not-a-real-checkpoint")
    labels_path.write_text(json.dumps(LABEL_MAP), encoding="utf-8")

    class BrokenTorch:
        @staticmethod
        def load(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise load_error

    assert (
        create_fra_artifact_predictor(
            model_path,
            labels_path,
            torch_module=BrokenTorch(),
        )
        is None
    )


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)

    def unsqueeze(self, axis: int) -> "FakeTensor":
        return FakeTensor(np.expand_dims(self.values, axis))

    def __getitem__(self, index: int) -> "FakeTensor":
        return FakeTensor(self.values[index])

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeModel:
    def __init__(self) -> None:
        self.loaded_state: tuple[object, bool] | None = None
        self.device: str | None = None
        self.eval_called = False
        self.input_shape: tuple[int, ...] | None = None

    def load_state_dict(self, state_dict: object, *, strict: bool) -> None:
        self.loaded_state = (state_dict, strict)

    def to(self, device: str) -> "FakeModel":
        self.device = device
        return self

    def eval(self) -> "FakeModel":
        self.eval_called = True
        return self

    def __call__(self, inputs: FakeTensor) -> FakeTensor:
        self.input_shape = inputs.values.shape
        return FakeTensor(np.array([[0.1, 3.0, 0.2, 0.0, -1.0]], dtype=np.float32))


class FakeTorch:
    def __init__(self) -> None:
        self.load_call: tuple[Path, dict[str, object]] | None = None

    def load(self, path: Path, **kwargs: object) -> dict[str, object]:
        self.load_call = (path, kwargs)
        return {"net.0.weight": object()}

    @staticmethod
    def from_numpy(values: np.ndarray) -> FakeTensor:
        return FakeTensor(values)

    @staticmethod
    def softmax(logits: FakeTensor, *, dim: int) -> FakeTensor:
        assert dim == 1
        del logits
        return FakeTensor(np.array([[0.05, 0.8, 0.05, 0.04, 0.06]], dtype=np.float32))

    @staticmethod
    def inference_mode() -> nullcontext[None]:
        return nullcontext()


def test_artifact_predictor_loads_safely_and_returns_normalized_diagnosis(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "fra_cnn_rocm.pt"
    labels_path = tmp_path / "fra_label_map.json"
    model_path.write_bytes(b"artifact")
    labels_path.write_text(json.dumps(LABEL_MAP), encoding="utf-8")
    torch = FakeTorch()
    model = FakeModel()

    predictor = create_fra_artifact_predictor(
        model_path,
        labels_path,
        torch_module=torch,
        model_factory=lambda _torch, _channels, _classes: model,
    )

    assert predictor is not None
    result = predictor(make_fra_frame())
    assert torch.load_call == (
        model_path,
        {"map_location": "cpu", "weights_only": True},
    )
    assert model.loaded_state == ({"net.0.weight": model.loaded_state[0]["net.0.weight"]}, True)
    assert model.device == "cpu"
    assert model.eval_called is True
    assert model.input_shape == (1, 3, 256)
    assert result.analysis_method == "fra_model_artifact"
    assert result.fault_class == "winding_deformation_suspected"
    assert result.is_anomalous is True
    assert result.confidence == pytest.approx(0.8)
    assert result.anomaly_score == pytest.approx(0.95)
    assert result.requires_human_review is True
    assert result.likely_faults == ["winding_deformation_suspected"]
    assert any("AMD-trained FRA artifact" in item for item in result.evidence)


class RecordedLayer:
    def __init__(self, kind: str, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        self.kind = kind
        self.args = args
        self.kwargs = kwargs


def layer_factory(kind: str):
    def create(*args: object, **kwargs: object) -> RecordedLayer:
        return RecordedLayer(kind, args, kwargs)

    return create


class FakeModule:
    def __init__(self) -> None:
        pass


class FakeSequential:
    def __init__(self, *layers: RecordedLayer) -> None:
        self.layers = layers


class FakeNN:
    Module = FakeModule
    Sequential = FakeSequential
    Conv1d = staticmethod(layer_factory("Conv1d"))
    ReLU = staticmethod(layer_factory("ReLU"))
    MaxPool1d = staticmethod(layer_factory("MaxPool1d"))
    AdaptiveAvgPool1d = staticmethod(layer_factory("AdaptiveAvgPool1d"))
    Linear = staticmethod(layer_factory("Linear"))


def test_fra_1d_cnn_uses_the_hackathon_architecture() -> None:
    model = build_fra_1d_cnn(SimpleNamespace(nn=FakeNN), in_channels=3, num_classes=5)

    assert [layer.kind for layer in model.net.layers] == [
        "Conv1d",
        "ReLU",
        "MaxPool1d",
        "Conv1d",
        "ReLU",
        "MaxPool1d",
        "Conv1d",
        "ReLU",
        "AdaptiveAvgPool1d",
    ]
    assert model.net.layers[0].args == (3, 32)
    assert model.net.layers[0].kwargs == {"kernel_size": 7, "padding": 3}
    assert model.net.layers[3].args == (32, 64)
    assert model.net.layers[3].kwargs == {"kernel_size": 5, "padding": 2}
    assert model.net.layers[6].args == (64, 128)
    assert model.net.layers[6].kwargs == {"kernel_size": 3, "padding": 1}
    assert model.net.layers[8].args == (1,)
    assert model.head.kind == "Linear"
    assert model.head.args == (128, 5)
