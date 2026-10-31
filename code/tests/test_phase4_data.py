"""Phase 4 실데이터 통합 테스트. data/processed/에 Phase 1·2 결과가 있을 때만 돈다 (없으면 skip).

예측 파일 검사는 03_baselines.py를 실행해 data/processed/preds/가 있을 때만 돈다.
"""
import polars as pl
import pytest

from afp import baselines, evaluate, io, metrics, preds

pytestmark = pytest.mark.skipif(
    not (io.TRAJECTORIES_PATH.exists() and io.STEPS_PATH.exists() and io.SPLITS_PATH.exists()),
    reason="data/processed에 trajectories·steps·splits가 없음 (01_parse.py, 02_split.py 실행 필요)",
)


@pytest.fixture(scope="module")
def sets():
    return evaluate.load_eval_sets()


@pytest.fixture(scope="module")
def costs(sets):
    return evaluate.load_costs(sets)


def test_real_fixed_cap_values(sets, costs):
    all_trajs = pl.concat([t.select(metrics.TRAJ_COLUMNS) for t in sets.values()])
    curve = metrics.fixed_cap_curve(all_trajs, pl.concat(list(costs.values())), ks=[30, 35, 40, 50])
    got = {r["K"]: (round(r["lost_success"], 3), round(r["saved_steps"], 3)) for r in curve.iter_rows(named=True)}
    assert got == {30: (0.070, 0.294), 35: (0.049, 0.246), 40: (0.037, 0.208), 50: (0.022, 0.158)}  # Phase 3
    assert round(curve.filter(pl.col("K") == 35)["saved_tokens"][0], 3) == 0.422

    ts = metrics.fixed_cap_curve(sets["test_same"], costs["test_same"], ks=[35]).row(0, named=True)
    assert round(ts["lost_success"], 4) == 0.0447 and round(ts["saved_steps"], 4) == 0.2181
    # 닫힌 식 = 하네스 (N=35에서 진행 중 궤적 전부 p_fail=1)
    alive = metrics.running(sets["test_same"], 35)
    p = alive.select("traj_id", N=pl.lit(35), p_fail=pl.lit(1.0))
    h = metrics.cost_curve(sets["test_same"], p, 35, costs=costs["test_same"]).row(-1, named=True)
    for c in ("lost_success", "saved_steps", "saved_tokens"):
        assert ts[c] == pytest.approx(h[c], abs=1e-12)


def test_real_cap_selected_on_train_is_34(sets):
    k = metrics.select_cap(metrics.fixed_cap_curve(sets["train"], ks=range(1, 81)), evaluate.MAX_LOSS)
    assert k["K"] == 34
    assert round(k["lost_success"], 4) == 0.0471 and round(k["saved_steps"], 4) == 0.2506


def test_real_issue_text_unique_after_strip():
    texts = baselines.issue_table()
    assert texts.height == 3_591
    raw = io.load_trajectories(["instance_id", "issue_text"]).unique()
    hinted = raw.filter(pl.col("issue_text").str.contains(baselines.HINT, literal=True))
    assert hinted["instance_id"].n_unique() == 13
    clean = raw.filter(~pl.col("issue_text").str.contains(baselines.HINT, literal=True))
    pairs = hinted.join(clean, on="instance_id", suffix="_clean")
    assert pairs.height == 13
    assert all(baselines.strip_hint(a) == b for a, b in pairs.select("issue_text", "issue_text_clean").iter_rows())


PHASE4_MODELS = ("b0_err", "b0_editfail", "b1_static", "b1_n", "ref_oracle_loo", "ref_oracle_xmodel")


@pytest.mark.skipif(not all(preds.path(m).exists() for m in PHASE4_MODELS), reason="예측 파일 없음 (03_baselines.py 실행 필요)")
def test_real_preds_cover_running(sets):
    for m in PHASE4_MODELS:
        meta = preds.metadata(m)
        s = {k: sets[k] for k in meta["splits"]}
        counts = preds.validate(preds.read(m), s, meta["n_grid"])
        for name, by_n in counts.items():
            for n, c in by_n.items():
                assert c == metrics.running(sets[name], n).height, (m, name, n)
    # B1은 이슈 안에서 점수가 같다 → 하네스 이슈 내 AUROC 정확히 0.5
    for m in ("b1_static", "b1_n"):
        p = preds.read(m)
        for name, t in sets.items():
            for n in evaluate.N_GRID_ISSUE:
                wi = metrics.within_issue_auroc(metrics.align(t, evaluate.split_preds(p, t, n), n))
                assert abs(wi["pair_weighted"] - 0.5) < 1e-12, (m, name, n)
    assert preds.metadata("ref_oracle_xmodel")["splits"] == ["test_same", "test_cross"]
    with pytest.raises(ValueError, match="피처로 쓸 수 없다"):
        preds.load_feature("ref_oracle_loo")
