"""CPU ONNX Runtime adapter for the exported LAIA classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


INPUT_NAMES = ("joint_features", "motion_features", "bone_features", "node_valid")


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities.astype(np.float32)


class ONNXRuntimeClassifier:
    def __init__(self, model_path: str | Path, *, intra_op_threads: int | None = None) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        if intra_op_threads is not None:
            options.intra_op_num_threads = int(intra_op_threads)
            options.inter_op_num_threads = 1
        self.model_path = Path(model_path).resolve()
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        actual_inputs = tuple(item.name for item in self.session.get_inputs())
        if actual_inputs != INPUT_NAMES:
            raise RuntimeError(f"inputs ONNX inesperados: {actual_inputs}")
        outputs = tuple(item.name for item in self.session.get_outputs())
        if outputs != ("logits",):
            raise RuntimeError(f"outputs ONNX inesperados: {outputs}")

    def logits(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        feed: dict[str, np.ndarray] = {}
        for name in INPUT_NAMES:
            value = np.asarray(features[name])
            if name == "node_valid":
                value = value.astype(bool, copy=False)
            else:
                value = value.astype(np.float32, copy=False)
            if value.ndim == (2 if name == "node_valid" else 3):
                value = value[None]
            feed[name] = np.ascontiguousarray(value)
        logits = np.asarray(self.session.run(["logits"], feed)[0], dtype=np.float32)
        if logits.ndim != 2 or logits.shape[1] != 7 or not np.isfinite(logits).all():
            raise RuntimeError(f"salida ONNX inválida: {logits.shape}")
        return logits

    def predict(self, features: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        logits = self.logits(features)
        return logits, softmax(logits)

