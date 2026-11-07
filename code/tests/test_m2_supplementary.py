"""Phase 6.5 보조 실험 (05b_m2_supplementary.py): 사전 등록한 모델 정의 — 주 모델의 고른 설정에서 누적 집계만 뺀다."""
import importlib

import pytest

pytest.importorskip("torch")
from afp import lstm, preds  # noqa: E402

supp = importlib.import_module("05b_m2_supplementary")


def dev_summary(b1_cfg=None):
    a = lstm.copy_config(lstm.M2Config(), hidden=32, dropout=0.1)
    b = lstm.copy_config(lstm.M2Config(), hidden=64, dropout=0.3, use_b1=True) if b1_cfg is None else b1_cfg
    return {"selected": {"m2_lstm": {"cfg": a.as_dict()}, "m2_lstm_b1": {"cfg": b.as_dict()}},
            "ablation": {"m2x_nocum": {"config": "h32_d10_nocum"}}}


def test_supp_configs_only_drop_cumulative_inputs():
    s = dev_summary()
    cfgs = supp.supp_cfgs(s)
    assert set(cfgs) == set(supp.SUPP)
    for name, base in supp.SUPP.items():
        want = {**s["selected"][base]["cfg"], "use_cum": False}
        assert cfgs[name].as_dict() == want
    assert supp.m2.cfg_key(cfgs["m2x_nocum"]) == "h32_d10_nocum"  # 05_m2.py 절제 모델과 같은 캐시 키 (재학습 없음)
    assert cfgs["m2x_nocum_b1"].use_b1 and not cfgs["m2x_nocum"].use_b1


def test_supp_configs_refuse_mismatched_ablation():
    s = dev_summary()
    s["ablation"]["m2x_nocum"]["config"] = "h16_d10_nocum"
    with pytest.raises(AssertionError):
        supp.supp_cfgs(s)


def test_pairs_and_model_ids():
    assert {a for a, _ in supp.PAIRS} == set(supp.SUPP)
    assert {b for _, b in supp.PAIRS} == set(supp.SUPP.values()) | set(supp.m2.M1_PAIR.values())
    for m in supp.SUPP:
        preds.path(m)  # 예측 파일 이름 규칙 통과 (m2x_)
