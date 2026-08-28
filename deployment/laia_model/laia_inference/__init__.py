"""Standalone inference package for the LAIA PSKUS hand-washing classifier."""

from .features_v2 import POSE_FEATURE_SCHEMA, build_pose_features_v2

__all__ = ["POSE_FEATURE_SCHEMA", "build_pose_features_v2"]

