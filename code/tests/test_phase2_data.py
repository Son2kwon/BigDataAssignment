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


def test_real_splits_match_proposal_numbers():
    traj = io.load_trajectories(split.INPUT_COLUMNS)
    splits = pl.read_parquet(io.SPLITS_PATH)
    stats = split.validate_splits(traj, splits)
    assert stats["trajectories"] == 80_036
    assert stats["split_trajectories"] == {"train": 62_460, "test_same": 12_332, "test_cross": 5_244}
    assert stats["train_issues"] == 3_013 and stats["test_issues"] == 578
    assert stats["train_test_issue_overlap"] == 0 and stats["train_issues_crossing_folds"] == 0


def test_real_splits_are_reproducible_and_documented():
    traj = io.load_trajectories(split.INPUT_COLUMNS)
    assert split.make_splits(traj).equals(pl.read_parquet(io.SPLITS_PATH))
    meta = json.loads(pq.read_schema(io.SPLITS_PATH).metadata[b"afp.split"])
    assert meta["method"] == "StratifiedGroupKFold" and meta["n_folds"] == split.N_FOLDS
    assert meta["heldout_fold"] == split.HELDOUT_FOLD and meta["seed"] == io.SEED
    assert pq.read_schema(io.SPLITS_PATH).names == ["traj_id", "split", "fold"]
