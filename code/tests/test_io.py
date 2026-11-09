"""결과 요약 JSON 반올림 (Phase 5.0a)."""
import json
import math

import numpy as np

from afp import io


def test_round_floats_nested_and_types():
    obj = {"a": 0.12345678901234, "b": [1, 2.000000000049, {"c": np.float64(0.333333333333333)}], "d": True, "e": None,
           "f": "0.123456789012", "g": (np.float32(0.5), 3), "h": float("nan"), "i": float("inf")}
    r = io.round_floats(obj)
    assert r["a"] == 0.123456789
    assert r["b"][1] == 2.0 and r["b"][0] == 1 and isinstance(r["b"][0], int)
    assert r["b"][2]["c"] == 0.3333333333 and type(r["b"][2]["c"]) is float
    assert r["d"] is True and r["e"] is None and r["f"] == "0.123456789012"
    assert r["g"] == [0.5, 3]
    assert math.isnan(r["h"]) and r["i"] == float("inf")


def test_round_floats_removes_machine_noise():
    # PR #11: 머신 사이 τ 차이 <= 2.9e-12. 반올림하면 같은 값이 된다 (경계 근처가 아니면)
    a, b = 0.9133456789123, 0.9133456789123 + 2.9e-12
    assert a != b and io.round_floats(a) == io.round_floats(b)


def test_write_summary_format(tmp_path):
    p = tmp_path / "summary.json"
    out = io.write_summary(p, {"x": 1 / 3, "k": {"한글": 2}})
    text = p.read_text()
    assert text.endswith("}\n") and "한글" in text and '"x": 0.3333333333' in text
    assert json.loads(text) == out
    io.write_summary(p, {"x": 1.0}, trailing_newline=False)
    assert p.read_text().endswith("}")


def test_clear_owned_files_keeps_other_files(tmp_path):
    """04_m1.py SHAP 그림 정리: 자기 패턴 파일만 지우고 다른 스크립트 그림(auroc_by_n.png 등)은 남긴다."""
    d = tmp_path / "figures"
    io.clear_owned_files(d, ("shap_summary_*.png", "shap_dependence_*.png"))  # 폴더가 없으면 만든다
    assert d.is_dir() and not any(d.iterdir())
    for name in ("auroc_by_n.png", "shap_summary_m1_xgb_N5.png", "shap_dependence_m1_xgb_N5_n_edit_ok.png", "notes.txt"):
        (d / name).write_bytes(b"x")
    (d / "sub").mkdir()
    (d / "sub" / "shap_summary_x.png").write_bytes(b"x")  # 하위 폴더는 건드리지 않는다
    removed = io.clear_owned_files(d, ("shap_summary_*.png", "shap_dependence_*.png"))
    assert sorted(p.name for p in removed) == ["shap_dependence_m1_xgb_N5_n_edit_ok.png", "shap_summary_m1_xgb_N5.png"]
    assert sorted(p.name for p in d.iterdir()) == ["auroc_by_n.png", "notes.txt", "sub"]
    assert (d / "sub" / "shap_summary_x.png").exists()
