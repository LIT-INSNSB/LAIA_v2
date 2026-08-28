"""Optional PyTorch adapter used for parity and fallback inference."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .model import load_torch_model
from .runtime_onnx import softmax


class TorchClassifier:
    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model, self.payload = load_torch_model(checkpoint_path, device=self.device)

    def logits(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        tensors: dict[str, torch.Tensor] = {}
        for name in ("joint_features", "motion_features", "bone_features", "node_valid"):
            value = np.asarray(features[name])
            if value.ndim == (2 if name == "node_valid" else 3):
                value = value[None]
            tensor = torch.from_numpy(np.ascontiguousarray(value)).to(self.device)
            tensors[name] = tensor.bool() if name == "node_valid" else tensor.float()
        with torch.inference_mode():
            output = self.model(
                tensors["joint_features"],
                tensors["motion_features"],
                tensors["bone_features"],
                tensors["node_valid"],
            )
        return output.detach().cpu().numpy().astype(np.float32)

    def predict(self, features: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        logits = self.logits(features)
        return logits, softmax(logits)

