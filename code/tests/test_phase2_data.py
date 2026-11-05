"""실제 데이터 통합 테스트. data/processed/에 Phase 1·2 결과가 있을 때만 돈다 (없으면 skip)."""
import json

import polars as pl
import pyarrow.parquet as pq
import pytest

from afp import io, split

pytestmark = pytest.mark.skipif(
    not (io.TRAJECTORIES_PATH.exists() and io.SPLITS_PATH.exists()),
    reason="data/processed에 trajectories·splits가 없음 (01_parse.py, 02_split.py 실행 필요)",
)


def test_real_splits_are_70b_only_random_issue_split():
    traj = io.load_trajectories(split.INPUT_COLUMNS)
    splits = pl.read_parquet(io.SPLITS_PATH)
    stats = split.validate_splits(traj, splits)
    assert traj.height == 80_036  # 제안서 전체 궤적 수
    assert stats["trajectories"] == 74_792  # 70B만 (제안서 모델별 수)
    assert stats["excluded_trajectories"] == {"swe-agent-llama-8b": 4_053, "swe-agent-llama-405b": 1_191}
    assert stats["split_trajectories"] == {"train": 62_311, "test": 12_481}
    assert stats["train_issues"] == 2_989 and stats["test_issues"] == 602
    assert stats["train_test_issue_overlap"] == 0 and stats["train_issues_crossing_folds"] == 0
    # 테스트 이슈는 8B·405B 시도 이슈(예전 테스트 578개)와 무관하게 뽑혔다
    other = set(traj.filter(pl.col("model_name") != split.TRAIN_MODEL)["instance_id"])
    test = set(traj.join(splits.filter(pl.col("split") == "test"), on="traj_id")["instance_id"])
    assert len(other) == 578 and test != other
    rate = traj.join(splits, on="traj_id").group_by("split").agg(pl.col("target").mean())
    assert rate["target"].max() - rate["target"].min() < 0.001  # 층화: 성공률 16.66% vs 16.70%


def test_real_splits_are_reproducible_and_documented():
    traj = io.load_trajectories(split.INPUT_COLUMNS)
    assert split.make_splits(traj).equals(pl.read_parquet(io.SPLITS_PATH))
    meta = json.loads(pq.read_schema(io.SPLITS_PATH).metadata[b"afp.split"])
    assert meta["method"] == "StratifiedGroupKFold" and meta["n_folds"] == split.N_FOLDS
    assert meta["heldout_fold"] == split.HELDOUT_FOLD and meta["seed"] == io.SEED
    assert meta["test_selection"]["n_splits"] == split.TEST_PARTS and meta["excluded_models"] == list(split.EXCLUDED_MODELS)
    assert pq.read_schema(io.SPLITS_PATH).names == ["traj_id", "split", "fold"]
