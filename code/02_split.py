"""Phase 2: 학습/테스트 분할, 학습 이슈 fold, 누수 점검

실행: uv run python code/02_split.py [--force] [--leak-sample 4000] [--skip-leakage]

입력: data/processed/trajectories.parquet, steps.parquet (01_parse.py), 원본 parquet (출처 확인·사후 누수 점검)
- data/processed/splits.parquet  traj_id, split ∈ {train, test_same, test_cross}, fold (train 0..4, 테스트 -1)
- results/phase2/                검증 수치, fold 균형, 학습/테스트 이슈 비교, 누수 점검 결과

splits.parquet가 이미 있으면 다시 계산한 결과와 같은지 확인한다. 다르면 멈추고, --force면 덮어쓴다.
끝에 제안서 수치와 대조하고 다르면 assert로 멈춘다.
"""
import argparse
import json
import sys

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import sklearn

from afp import io, leakage, split

EXPECTED = {
    "trajectories": 80_036,
    "split_trajectories": {"train": 62_460, "test_same": 12_332, "test_cross": 5_244},
    "train_issues": 3_013,
    "test_issues": 578,
    "train_test_issue_overlap": 0,
    "train_issues_crossing_folds": 0,
}
# 분할·비교에 읽는 궤적 컬럼 (exit_status 제외)
TRAJ_COLUMNS = ["traj_id", "file_idx", "row_idx", "instance_id", "model_name", "target", "n_steps", "system_hash"]
EARLY_STEPS = 20
N_EXAMPLES = 5


def check_provenance(traj: pl.DataFrame) -> dict:
    """궤적 표가 download.sh로 고정한 원본 12개와 행 단위로 맞는지 확인한다."""
    sha = io.verify_raw_checksums()
    expected_id = pl.format("{}-{}", pl.col("file_idx").cast(pl.String).str.zfill(2), pl.col("row_idx").cast(pl.String).str.zfill(5))
    if traj.filter(pl.col("traj_id") != expected_id).height:
        raise ValueError("traj_id가 file_idx·row_idx 규칙과 다르다")
    raw = pl.concat([
        pl.from_arrow(io.read_raw(k, ["instance_id", "model_name", "target"])).with_row_index("row_idx").with_columns(file_idx=pl.lit(k))
        for k in range(io.N_RAW_FILES)
    ])
    joined = traj.join(raw.with_columns(pl.col("row_idx").cast(pl.Int32), pl.col("file_idx").cast(pl.Int8)),
                       on=["file_idx", "row_idx"], how="full", suffix="_raw")
    bad = joined.filter(
        pl.col("traj_id").is_null() | pl.col("instance_id_raw").is_null()
        | (pl.col("instance_id") != pl.col("instance_id_raw"))
        | (pl.col("model_name") != pl.col("model_name_raw"))
        | (pl.col("target") != pl.col("target_raw"))
    ).height
    if bad:
        raise ValueError(f"궤적 표와 원본이 다른 행 {bad}개")
    return {"raw_rows": raw.height, "raw_sha256_verified": len(sha), "trajectory_rows_matching_raw": traj.height}


def build_splits(traj: pl.DataFrame, force: bool) -> tuple[pl.DataFrame, dict]:
    splits = split.make_splits(traj)
    # 입력 순서를 섞어도 같은 결과인지 (결정성)
    shuffled = traj.sample(fraction=1.0, shuffle=True, seed=io.SEED + 1)
    if not split.make_splits(shuffled).equals(splits):
        raise AssertionError("입력 순서에 따라 분할이 달라진다")
    det = {"same_result_on_shuffled_input": True}
    if io.SPLITS_PATH.exists():
        old = pl.read_parquet(io.SPLITS_PATH)
        same = old.equals(splits)
        det["matches_existing_file"] = same
        if not same and not force:
            raise SystemExit(f"{io.SPLITS_PATH}가 다시 계산한 분할과 다르다. 확인 후 --force로 덮어쓴다.")
        if same:
            return splits, det
    meta = {
        "method": "StratifiedGroupKFold", "n_folds": split.N_FOLDS, "shuffle": True, "seed": io.SEED,
        "group": "instance_id", "stratify": "target", "heldout_fold": split.HELDOUT_FOLD,
        "train_model": split.TRAIN_MODEL, "heldout_models": list(split.HELDOUT_MODELS),
        "sklearn": sklearn.__version__,
    }
    table = splits.to_arrow()
    table = table.replace_schema_metadata({b"afp.split": json.dumps(meta).encode()})
    tmp = io.SPLITS_PATH.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(io.SPLITS_PATH)
    det["written"] = True
    return splits, det


def fold_diagnostics(traj: pl.DataFrame, splits: pl.DataFrame, out_dir) -> dict:
    train = traj.join(splits, on="traj_id").filter(pl.col("split") == "train").sort("traj_id")
    tables, stats = [], {}
    for name, stratified in (("GroupKFold", False), ("StratifiedGroupKFold", True)):
        fold = split.train_folds(train, stratified=stratified)
        if stratified and not np.array_equal(fold, train["fold"].to_numpy()):
            raise AssertionError("저장한 fold와 StratifiedGroupKFold 재계산이 다르다")
        t = split.fold_balance(train, fold).with_columns(method=pl.lit(name))
        tables.append(t)
        stats[name] = {
            "success_rate_min": round(t["success_rate"].min(), 4),
            "success_rate_max": round(t["success_rate"].max(), 4),
            "success_rate_range": round(t["success_rate"].max() - t["success_rate"].min(), 4),
            "n_traj_range": int(t["n_traj"].max() - t["n_traj"].min()),
            "n_issues_range": int(t["n_issues"].max() - t["n_issues"].min()),
        }
    pl.concat(tables).select("method", "fold", "n_traj", "n_issues", "success_rate", "issues_with_success") \
        .write_csv(out_dir / "fold_balance.csv", float_precision=4)
    stats["chosen"] = "StratifiedGroupKFold"
    return stats


def split_tables(traj: pl.DataFrame, splits: pl.DataFrame, out_dir) -> None:
    df = traj.join(splits, on="traj_id")
    order = {s: i for i, s in enumerate(split.SPLITS)}
    df.group_by("split", "model_name").agg(
        n_traj=pl.len(), n_issues=pl.col("instance_id").n_unique(),
        success_rate=pl.col("target").mean(), n_steps_median=pl.col("n_steps").median(),
    ).sort(pl.col("split").replace_strict(order), "model_name").write_csv(out_dir / "split_counts.csv", float_precision=4)


def leakage_static() -> dict:
    """(a) 누수 컬럼이 피처 쪽 입력에 들어갈 길이 없는지."""
    refs = leakage.scan_code()
    outside = {f: h for f, h in refs.items() if f not in leakage.LEAKY_ALLOWLIST}
    processed = {p.name: pq.read_schema(p).names for p in sorted(io.PROCESSED_DIR.glob("*.parquet"))}
    leaky_in_processed = {name: [c for c in cols if c in io.LEAKY_COLUMNS] for name, cols in processed.items()}
    try:
        io.load_trajectories(["traj_id", *io.LEAKY_COLUMNS])
        guard = False
    except ValueError:
        guard = True
    return {
        "files_referencing_leaky_columns": {f: sorted({n for _, n in h}) for f, h in refs.items()},
        "references_outside_allowlist": outside,
        "allowlist": list(leakage.LEAKY_ALLOWLIST),
        "leaky_columns_in_processed": {k: v for k, v in leaky_in_processed.items() if v},
        "splits_columns": processed.get(io.SPLITS_PATH.name),
        "safe_loader_rejects_leaky": guard,
    }


def marker_scan() -> dict:
    """전체 궤적의 앞 20스텝 관측에서 하네스 문자열을 센다 (표본이 아니라 전수)."""
    obs = pl.scan_parquet(io.STEPS_PATH).filter(pl.col("step_idx") <= EARLY_STEPS).select("traj_id", "obs_text")
    exprs = [pl.col("obs_text").str.contains(m, literal=True).fill_null(False).alias(m)
             for m in (*leakage.HARNESS_MARKERS, leakage.HARNESS_DIR)]
    flags = obs.select("traj_id", *exprs).collect()
    out = {"observations": flags.height}
    for m in (*leakage.HARNESS_MARKERS, leakage.HARNESS_DIR):
        hit = flags.filter(pl.col(m))
        out[m] = {"observations": hit.height, "trajectories": hit["traj_id"].n_unique()}
    return out


def _snippet(text: str, needle: str, width: int = 160) -> str:
    i = text.find(needle)
    if i < 0:
        return text[:2 * width]
    return text[max(0, i - width): i + len(needle) + width]


def leakage_overlap(traj: pl.DataFrame, n_sample: int, out_dir) -> dict:
    """(b) eval_logs 테스트 식별자 vs 앞 20스텝 관측 (층화 표본)."""
    sample = leakage.sample_trajectories(io.load_trajectories(leakage.SAMPLE_COLUMNS), n_sample)
    logs = {tid: leakage.parse_eval_log(text) for tid, text in leakage.load_eval_logs(sample).items()}
    steps = (
        io.scan_steps(leakage.STEP_COLUMNS)
        .filter(pl.col("traj_id").is_in(sample["traj_id"].to_list()) & (pl.col("step_idx") <= EARLY_STEPS))
        .sort("traj_id", "step_idx").collect()
    )
    steps_by = {k[0]: g.sort("step_idx").to_dicts() for k, g in steps.group_by("traj_id")}
    issue = dict(zip(sample["traj_id"], sample["issue_text"]))
    repo_of = dict(zip(sample["traj_id"], sample.select(split.repo_of(pl.col("instance_id")))["instance_id"]))
    generic = leakage.generic_names(logs, repo_of)

    per_traj, matches = [], []
    for r in sample.iter_rows(named=True):
        tid = r["traj_id"]
        summ, rows = leakage.audit_trajectory(logs[tid], issue[tid], steps_by.get(tid, []), generic)
        per_traj.append({"traj_id": tid, "model_name": r["model_name"], "target": r["target"], "group": "observed", **summ})
        for m in rows:
            matches.append({"traj_id": tid, "target": r["target"], **m})

    # 대조군: 다른 이슈의 식별자를 붙였을 때 겹치는 비율
    with_tests = sample.filter(pl.col("traj_id").is_in([t for t, lg in logs.items() if lg.tests]))
    meta_of = {r["traj_id"]: r for r in with_tests.iter_rows(named=True)}
    for group, same_repo in (("control_same_repo", True), ("control_other_repo", False)):
        for tid, donor in leakage.control_donors(with_tests, same_repo).items():
            r = meta_of[tid]
            summ, _ = leakage.audit_trajectory(logs[donor], issue[tid], steps_by.get(tid, []), generic)
            per_traj.append({"traj_id": tid, "model_name": r["model_name"], "target": r["target"], "group": group, **summ})

    pt = pl.DataFrame(per_traj, infer_schema_length=None)
    mt = pl.DataFrame(matches, infer_schema_length=None) if matches else None

    def rates(d: pl.DataFrame) -> dict:
        t = d.filter(pl.col("n_test_names") > 0)
        return {
            "n_traj": d.height,
            "n_with_test_ids": t.height,
            "any_test_name_hit": t.filter(pl.col("name_hits") > 0).height,
            "any_specific_name_hit": t.filter(pl.col("specific_name_hits") > 0).height,
            "any_specific_name_hit_unexplained": t.filter(pl.col("specific_name_hits_unexplained") > 0).height,
            "any_node_id_hit": t.filter(pl.col("node_hits") > 0).height,
            "any_test_file_hit": d.filter(pl.col("file_hits") > 0).height,
            "any_failed_test_name_hit": t.filter(pl.col("failed_name_hits") > 0).height,
            "with_new_test_files": d.filter(pl.col("n_new_test_files") > 0).height,
            "any_new_test_file_hit": d.filter(pl.col("new_file_hits") > 0).height,
            "any_new_file_test_name_hit": t.filter(pl.col("new_name_hits") > 0).height,
            "any_result_line": t.filter(pl.col("result_line_hits") > 0).height,
            "any_result_line_without_test_run": t.filter(pl.col("result_line_without_test_run") > 0).height,
            "any_harness_marker": d.filter(pl.col("harness_marker_steps") > 0).height,
            "any_harness_dir": d.filter(pl.col("harness_dir_steps") > 0).height,
            "test_name_share_matched_mean": round(float((t["name_hits"] / t["n_test_names"]).mean() or 0), 4),
            **{f"cat_{c}": int(d[f"cat_{c}"].sum()) for c in leakage.CATEGORIES},
        }

    rows = []
    obs = pt.filter(pl.col("group") == "observed")
    rows.append({"group": "observed", "subset": "all", **rates(obs)})
    for model in split.KNOWN_MODELS:
        rows.append({"group": "observed", "subset": model, **rates(obs.filter(pl.col("model_name") == model))})
    for target in (True, False):
        rows.append({"group": "observed", "subset": f"target={target}", **rates(obs.filter(pl.col("target") == target))})
    for group in ("control_same_repo", "control_other_repo"):
        rows.append({"group": group, "subset": "all", **rates(pt.filter(pl.col("group") == group))})
    pl.DataFrame(rows).write_csv(out_dir / "leakage_overlap.csv")

    status = obs["eval_status"].value_counts().sort(["count", "eval_status"], descending=[True, False])
    if obs.filter(pl.col("n_test_names") > 0).height < 0.5 * obs.height:
        raise AssertionError("표본의 절반 이상에서 테스트 식별자를 꺼내지 못했다. 파서를 확인한다")

    # 대표 예시: 범주별, 그리고 강한 신호(결과 줄, 새 테스트 파일, 하네스 문자열)는 가능한 만큼
    examples = []
    if mt is not None:
        text_of = {(r["traj_id"], r["step_idx"]): r for r in steps.iter_rows(named=True)}
        picks = [mt.filter(pl.col("category") == c).head(N_EXAMPLES) for c in leakage.CATEGORIES]
        picks.append(mt.filter(~pl.col("result_line_command_is_test_run")).head(3 * N_EXAMPLES))
        picks.append(mt.filter(pl.col("kind") == "new_test_file").head(3 * N_EXAMPLES))
        picks.append(mt.filter(pl.col("in_new_test_file") & (pl.col("kind") == "test_name")).head(3 * N_EXAMPLES))
        for row in pl.concat(picks).unique(["traj_id", "identifier"], maintain_order=True).iter_rows(named=True):
            st = text_of[(row["traj_id"], row["first_step"])]
            examples.append({**row, "obs_snippet": _snippet(st["obs_text"] or "", row["identifier"])})
        with (out_dir / "leakage_examples.jsonl").open("w") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        mt.group_by("kind", "category", "generic_name").agg(
            n=pl.len(), trajectories=pl.col("traj_id").n_unique(),
            with_result_line=pl.col("result_line_step").is_not_null().sum(),
            failed_in_eval=pl.col("failed_in_eval").sum(),
        ).sort("kind", "generic_name", "n", descending=[False, False, True]).write_csv(out_dir / "leakage_categories.csv")

    return {
        "sample": {"n": sample.height, "method": "(model_name, target) 층화 무작위, sklearn train_test_split", "seed": io.SEED,
                   "by_model": dict(sample["model_name"].value_counts().sort("model_name").iter_rows()),
                   "success_rate": round(sample["target"].mean(), 4)},
        "eval_status": dict(status.iter_rows()),
        "tests_parsed_total": int(obs["n_tests"].sum()),
        "generic_test_names": {"n": len(generic), "rule": "표본 채점 로그에서 3개 이상 저장소에 나오거나 5자 미만"},
        "overlap": rows,
        "examples_saved": len(examples),
    }


def check(traj, splits, stats, det, prov, folds, static, markers, overlap, out_dir):
    hashes = traj["system_hash"].value_counts().sort(["count", "system_hash"], descending=[True, False])
    s = {
        "provenance": prov,
        "determinism": det,
        "splits": stats,
        "folds": folds,
        "system_prompt": {"n_hashes": hashes.height, "hashes": hashes.to_dicts()},
        "leakage_static": static,
        "harness_marker_scan_first20": markers,
        "leakage_eval_overlap_first20": overlap,
    }
    (out_dir / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({k: v for k, v in s.items() if k != "leakage_eval_overlap_first20"}, ensure_ascii=False, indent=2, default=str))
    if overlap:
        print(pl.read_csv(out_dir / "leakage_overlap.csv"))

    for k, v in EXPECTED.items():
        assert stats[k] == v, (k, stats[k], v)
    assert hashes.height == 1, "시스템 프롬프트 해시가 여러 개"
    assert not static["references_outside_allowlist"], static["references_outside_allowlist"]
    assert not static["leaky_columns_in_processed"].get(io.SPLITS_PATH.name)
    assert static["splits_columns"] == list(split.SPLIT_SCHEMA), static["splits_columns"]
    assert static["safe_loader_rejects_leaky"]
    assert folds["StratifiedGroupKFold"]["success_rate_range"] < folds["GroupKFold"]["success_rate_range"]
    print("\n완료 기준 통과")


def main():
    ap = argparse.ArgumentParser(description="Phase 2: 분할·fold·누수 점검")
    ap.add_argument("--force", action="store_true", help="저장된 splits.parquet와 달라도 덮어씀")
    ap.add_argument("--leak-sample", type=int, default=4000, help="채점 로그 테스트 식별자 겹침 점검(2.3 b) 표본 크기 (기본 4000)")
    ap.add_argument("--skip-leakage", action="store_true", help="채점 로그 겹침 점검(2.3 b)을 건너뜀 (분할·정적 점검만)")
    args = ap.parse_args()
    if args.leak_sample <= 0:
        ap.error("--leak-sample은 양수여야 한다")
    for p in (io.TRAJECTORIES_PATH, io.STEPS_PATH):
        if not p.exists():
            sys.exit(f"{p}가 없다. 먼저 uv run python code/01_parse.py 를 실행한다.")

    out_dir = io.RESULTS_DIR / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj = io.load_trajectories(TRAJ_COLUMNS)
    split.validate_input(traj)
    prov = check_provenance(traj)
    splits, det = build_splits(traj, args.force)
    stats = split.validate_splits(traj, pl.read_parquet(io.SPLITS_PATH))
    folds = fold_diagnostics(traj, splits, out_dir)
    split_tables(traj, splits, out_dir)
    metrics, repos = split.compare_issue_sets(traj, splits)
    metrics.write_csv(out_dir / "issue_comparison.csv", float_precision=4)
    repos.write_csv(out_dir / "repo_composition.csv", float_precision=4)
    static = leakage_static()
    markers = marker_scan()
    overlap = None if args.skip_leakage else leakage_overlap(traj, args.leak_sample, out_dir)
    check(traj, splits, stats, det, prov, folds, static, markers, overlap, out_dir)


if __name__ == "__main__":
    main()
