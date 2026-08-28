"""Exact inference architecture used by the selected LightSTGCNV2 checkpoint.

This module intentionally contains no trainer, optimizer, scheduler, dataset,
or W&B dependency. PyTorch is optional at deployment time and is needed only
for source-checkpoint parity or the fallback PyTorch runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MEDIAPIPE_HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def _normalize_digraph(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=0)
    inverse = np.zeros_like(degree, dtype=np.float32)
    inverse[degree > 0] = 1.0 / degree[degree > 0]
    return adjacency @ np.diag(inverse)


def build_two_hand_adjacency() -> torch.Tensor:
    """Build the exact three-partition graph with corresponding-hand edges."""
    nodes = 42
    inward_edges: list[tuple[int, int]] = []
    for offset in (0, 21):
        inward_edges.extend((source + offset, target + offset) for source, target in MEDIAPIPE_HAND_EDGES)
    for node in range(21):
        inward_edges.extend(((node, node + 21), (node + 21, node)))
    identity = np.eye(nodes, dtype=np.float32)
    inward = np.zeros((nodes, nodes), dtype=np.float32)
    for source, target in inward_edges:
        inward[target, source] = 1.0
    outward = inward.T.copy()
    return torch.from_numpy(
        np.stack([identity, _normalize_digraph(inward), _normalize_digraph(outward)], axis=0)
    )


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, partitions: int) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.partitions = int(partitions)
        self.projection = nn.Conv2d(in_channels, out_channels * partitions, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, _, time, nodes = x.shape
        projected = self.projection(x).view(
            batch, self.partitions, self.out_channels, time, nodes
        )
        return torch.einsum("bkctv,kvw->bctw", projected, adjacency)


class STGCBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        partitions: int,
        temporal_kernel_size: int,
        stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (temporal_kernel_size - 1) // 2
        self.stride = int(stride)
        self.graph = SpatialGraphConv(in_channels, out_channels, partitions)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if in_channels == out_channels and stride == 1:
            self.residual: nn.Module = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.temporal(self.graph(x, adjacency)) + self.residual(x)
        output = self.activation(output)
        if self.stride > 1:
            mask = F.max_pool2d(
                mask,
                kernel_size=(self.stride, 1),
                stride=(self.stride, 1),
                ceil_mode=True,
            )
        output = output * mask.to(dtype=output.dtype)
        return output, mask


class PoseStreamStem(nn.Module):
    def __init__(self, out_channels: int = 16) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(3, int(out_channels), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(out_channels)),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        stream_mask = (features[:, 2:3] > 0.5).to(dtype=features.dtype)
        return self.projection(features * stream_mask) * stream_mask


class LightSTGCNV2(nn.Module):
    """Three-stream 42-node temporal graph classifier used in LAIA."""

    def __init__(
        self,
        *,
        num_classes: int = 7,
        stream_stem_channels: int = 16,
        channels: tuple[int, ...] = (64, 96, 128, 128),
        temporal_kernel_size: int = 9,
        temporal_strides: tuple[int, ...] = (1, 1, 2, 1),
        classifier_hidden_channels: int = 96,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        adjacency = build_two_hand_adjacency()
        self.register_buffer("adjacency", adjacency, persistent=True)
        self.joint_stem = PoseStreamStem(stream_stem_channels)
        self.motion_stem = PoseStreamStem(stream_stem_channels)
        self.bone_stem = PoseStreamStem(stream_stem_channels)
        previous = int(stream_stem_channels) * 3
        blocks: list[nn.Module] = []
        for output, stride in zip(channels, temporal_strides):
            blocks.append(
                STGCBlock(
                    previous,
                    int(output),
                    partitions=int(adjacency.shape[0]),
                    temporal_kernel_size=int(temporal_kernel_size),
                    stride=int(stride),
                    dropout=float(dropout),
                )
            )
            previous = int(output)
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], int(classifier_hidden_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(classifier_hidden_channels), int(num_classes)),
        )

    def forward(
        self,
        joint_features: torch.Tensor,
        motion_features: torch.Tensor,
        bone_features: torch.Tensor,
        node_valid: torch.Tensor,
    ) -> torch.Tensor:
        encoded = (
            self.joint_stem(joint_features),
            self.motion_stem(motion_features),
            self.bone_stem(bone_features),
        )
        x = torch.cat(encoded, dim=1)
        mask = node_valid[:, None].to(dtype=x.dtype)
        x = x * mask
        for block in self.blocks:
            x, mask = block(x, self.adjacency, mask)
        denominator = mask.sum(dim=(2, 3)).clamp_min(1.0)
        pooled = (x * mask).sum(dim=(2, 3)) / denominator
        return self.classifier(pooled)


def build_model() -> LightSTGCNV2:
    return LightSTGCNV2()


def extract_state_dict(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    if "model_state_dict" in payload:
        return payload["model_state_dict"]
    if "state_dict" in payload:
        return payload["state_dict"]
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload  # type: ignore[return-value]
    raise ValueError("checkpoint sin model_state_dict/state_dict compatible")


def load_torch_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[LightSTGCNV2, Mapping[str, Any]]:
    payload = torch.load(Path(checkpoint_path), map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint no es un mapping")
    model = build_model().to(device)
    model.load_state_dict(extract_state_dict(payload), strict=True)
    model.eval()
    return model, payload

