"""M1 피처 누수 테스트 (Phase 5.2): N 뒤 스텝, 금지 컬럼, 실데이터 1:1·결측."""
import polars as pl
import pytest

from afp import baselines, evaluate, features, io, metrics
from feature_helpers import features_of, random_steps


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_features_ignore_steps_after_n(seed):
    """5.2 누수 테스트: N 뒤의 스텝을 바꾸거나 지우거나 덧붙여도 시점 N 피처가 같다 (test_error_count_ignores_steps_after_n 확장)."""
    steps, trajs, rng = random_steps(seed)
    for n in (3, 5, 8, 10):
        base = features_of(steps, trajs, [n])
        after = pl.col("step_idx") > n
        changed = steps.with_columns(
            command=pl.when(after).then(pl.lit("submit")).otherwise(pl.col("command")),
            command_name=pl.when(after).then(pl.lit("submit")).otherwise(pl.col("command_name")),
            obs_len=pl.when(after).then(pl.lit(10 ** 6)).otherwise(pl.col("obs_len")),
            is_error=pl.when(after).then(pl.lit(True)).otherwise(pl.col("is_error")),
            edit_failed=pl.when(after).then(pl.lit(True)).otherwise(pl.col("edit_failed")),
            open_file=pl.when(after).then(pl.lit("/r/pkg/f0.py")).otherwise(pl.col("open_file")),
        )
        removed = steps.filter(~after)
        extra = pl.concat([steps, steps.filter(pl.col("step_idx") == 1).with_columns(step_idx=pl.lit(999, dtype=pl.Int64))])
        for variant in (changed, removed, extra, steps.sample(fraction=1.0, shuffle=True, seed=seed)):  # 행 순서도 무관
            assert features_of(variant, trajs, [n]).equals(base), n


def test_features_ignore_n_steps_beyond_running():
    """궤적 길이는 "N에 진행 중인가"(n_steps > N)만 정하고 피처 값에는 들어가지 않는다."""
    steps, trajs, _ = random_steps(3)
    base = features_of(steps, trajs, [5])
    longer = trajs.with_columns(n_steps=pl.when(pl.col("n_steps") > 5).then(pl.col("n_steps") + 100).otherwise(pl.col("n_steps")))
    assert features_of(steps, longer, [5]).equals(base)


def test_features_without_leaky_columns():
    """5.2: 금지 컬럼(exit_status, generated_patch, eval_logs)이 없는 입력으로 계산되고, 있어도 읽지 않는다."""
    assert not set(io.LEAKY_COLUMNS) & set(features.STEP_INPUT_COLUMNS)
    io.check_safe_columns(features.STEP_INPUT_COLUMNS)  # 누수 컬럼이 있으면 ValueError
    steps, trajs, _ = random_steps(4)
    base = features_of(steps, trajs, [3, 5])
    poisoned_steps = steps.with_columns(exit_status=pl.lit("submitted"), generated_patch=pl.lit("diff --git a/pkg/f0.py"), eval_logs=pl.lit("PASSED"))
    poisoned_trajs = trajs.with_columns(exit_status=pl.lit("early_exit"), target=pl.lit(True))
    assert features_of(poisoned_steps, poisoned_trajs, [3, 5]).equals(base)


def test_issue_mentions_ignore_gold_file_hint():
    """이슈 13개 끝의 "고칠 파일 목록"(정답 패치 정보)은 언급으로 쓰지 않는다 (B1과 같은 strip_hint)."""
    body = "Crash in pkg/core.py when x is None"
    hinted = body + "\n\n\n" + baselines.HINT + " in order to fix the issue.\nFILES TO MODIFY:\n- pkg/secret_fix.py\n- other/util.py"
    assert features.issue_file_mentions(hinted) == features.issue_file_mentions(body) == ["pkg/core.py"]


# ---- 실데이터 (데이터가 있을 때만) ----

@pytest.mark.skipif(not (io.STEPS_PATH.exists() and io.SPLITS_PATH.exists()), reason="steps.parquet / splits.parquet 없음")
def test_real_features_one_to_one_and_ignore_later_steps():
    """5.2 실데이터: 모든 N에서 진행 중 궤적과 1:1, 결측·무한값 없음. 표본 궤적은 앞 N스텝만 넣어 계산해도 같다."""
    sets = evaluate.load_eval_sets()
    trajs = pl.concat([t.select("traj_id", "instance_id", "n_steps") for t in sets.values()])
    values = features.load_step_values(max(evaluate.N_GRID), trajs)
    t = io.load_trajectories(["traj_id", "instance_id", "issue_text"]).join(trajs.select("traj_id"), on="traj_id", how="semi")
    m = features.mentions_table(t)
    feats = features.m1_features(values, trajs, evaluate.N_GRID, m)
    counts = features.check_features(feats, trajs, evaluate.N_GRID)
    assert counts == {n: metrics.running(trajs, n).height for n in evaluate.N_GRID}
    sample = trajs.filter(pl.col("n_steps") > 20).sample(n=300, seed=io.SEED)
    raw = io.scan_steps(features.STEP_INPUT_COLUMNS).join(sample.select("traj_id").lazy(), on="traj_id", how="semi").collect()
    ts = t.join(sample.select("traj_id"), on="traj_id", how="semi")
    ms = features.mentions_table(ts)
    for n in (3, 10):
        only_n = features.step_values(raw.filter(pl.col("step_idx") <= n), ms, ts)
        got = features.m1_features(only_n, sample, [n], ms)
        want = feats.join(sample.select("traj_id"), on="traj_id", how="semi").filter(pl.col("N") == n)
        assert got.equals(want)
    # 이슈 본문의 "고칠 파일 목록"이 언급 추출에 들어가지 않는다 (실데이터 13개 이슈)
    hinted = t.filter(pl.col("issue_text").str.contains(baselines.HINT, literal=True))
    assert hinted["instance_id"].n_unique() == 13
    for text in hinted.unique("instance_id")["issue_text"].to_list():
        assert features.issue_file_mentions(text) == features.issue_file_mentions(baselines.strip_hint(text))
