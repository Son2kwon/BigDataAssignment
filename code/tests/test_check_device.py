"""5b~5e 학습 장치 확인 (04e_m1_window.check_device, P5b-D15: cuda를 못 쓰면 CPU로 바꾸지 않고 멈춘다).

실제 xgboost를 import하지 않고 가짜 모듈을 넣는다. macOS에서 이 테스트가 xgboost를 먼저 올리면 뒤의 torch 테스트(test_lstm)가
세그폴트로 죽었다 (두 라이브러리의 OpenMP 런타임 충돌로 보임 [가정], 2026-11-15).
"""
import importlib
import sys
import types

import pytest

m1w = importlib.import_module("04e_m1_window")


def fake_xgboost(use_cuda):
    mod = types.ModuleType("xgboost")
    mod.__version__ = "fake"
    mod.build_info = lambda: {"USE_CUDA": use_cuda}

    class XGBClassifier:  # 빌드 정보로 먼저 멈추므로 학습까지 가지 않는다
        def __init__(self, **kw):
            raise AssertionError("빌드 확인 전에 학습하면 안 된다")
    mod.XGBClassifier = XGBClassifier
    return mod


def test_cuda_on_build_without_cuda_exits(monkeypatch):
    """CUDA 없이 빌드된 XGBoost는 device="cuda"를 조용히 CPU로 바꾸므로, 빌드 정보로 먼저 멈춘다."""
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgboost(False))
    with pytest.raises(SystemExit, match="CUDA 없이 빌드"):
        m1w.check_device("cuda")


def test_cpu_always_passes(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgboost(False))
    assert m1w.check_device("cpu") is None
