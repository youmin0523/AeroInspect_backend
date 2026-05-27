# =============================================
# tests/conftest.py
# 역할: 공통 pytest fixture 모음.
#   - ONNX_WEIGHTS_DIR: ONNX 가중치 디렉터리 경로 (env 로 override 가능)
# =============================================
from __future__ import annotations

import os
from pathlib import Path

import pytest


# 운영 backend 의 models_weights 는 비어있고, 실제 ONNX 는 통합 repo 의
# TEAM_PROJECT_2_Drone_project/backend/models_weights 에 있다.
# CI / 다른 환경에서는 ONNX_WEIGHTS_DIR 환경변수로 override.
_DEFAULT_WEIGHTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "TEAM_PROJECT_2_Drone_project"
    / "backend"
    / "models_weights"
)


@pytest.fixture(scope="session")
def onnx_weights_dir() -> Path:
    """ONNX 가중치 디렉터리. 환경변수 ONNX_WEIGHTS_DIR 가 우선."""
    env = os.environ.get("ONNX_WEIGHTS_DIR")
    if env:
        return Path(env)
    return _DEFAULT_WEIGHTS_DIR


@pytest.fixture(scope="session")
def datasets_dir() -> Path:
    """data.yaml 들이 있는 datasets 루트. 환경변수 DATASETS_DIR override."""
    env = os.environ.get("DATASETS_DIR")
    if env:
        return Path(env)
    return (
        Path(__file__).resolve().parents[2]
        / "TEAM_PROJECT_2_Drone_project"
        / "backend"
        / "training"
        / "datasets"
    )
