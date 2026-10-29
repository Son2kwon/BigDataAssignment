"""Phase 1: 원본 궤적 → trajectories.parquet, steps.parquet

실행: uv run python code/01_parse.py [--force]

- data/processed/trajectories.parquet  궤적당 1행
- data/processed/steps.parquet         스텝당 1행 (원본 파일 하나 = row group 묶음 하나)
- results/phase1/                      명령 빈도표, 검증 수치

결과 파일이 이미 있으면 파싱을 건너뛰고 검증만 한다. --force면 다시 만든다.
끝에 제안서 수치와 대조하고 다르면 assert로 멈춘다.
"""
import argparse
import collections
import json

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from afp import io, parse

TRAJ_SCHEMA = pa.schema([
    ("traj_id", pa.string()),
    ("file_idx", pa.int8()),
    ("row_idx", pa.int32()),
    ("instance_id", pa.string()),
    ("model_name", pa.string()),
    ("target", pa.bool_()),
    ("n_steps", pa.int16()),
    ("exit_status", pa.string()),  # 분석 전용 (라벨 누수)
    ("issue_text", pa.string()),
    ("issue_msg_len", pa.int32()),
    ("system_len", pa.int32()),
    ("system_hash", pa.string()),
    ("role_pattern_ok", pa.bool_()),
    ("roles_head", pa.string()),
])

STEP_SCHEMA = pa.schema([
    ("traj_id", pa.string()),
    ("step_idx", pa.int16()),
    ("action_text", pa.string()),
    ("command", pa.string()),
    ("command_head", pa.string()),
    ("command_name", pa.string()),
    ("obs_text", pa.string()),
    ("ai_len", pa.int32()),
    ("obs_len", pa.int32()),
    ("is_error", pa.bool_()),
    ("edit_failed", pa.bool_()),
    ("open_file", pa.string()),
    ("cwd", pa.string()),
])

# 원본에서 읽는 컬럼. generated_patch, eval_logs는 읽지 않는다.
RAW_COLUMNS = ["instance_id", "model_name", "target", "trajectory", "exit_status"]


def build(out_dir):
    io.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    head_counts = collections.Counter()
    traj_tables = []
    tmp_steps = io.STEPS_PATH.with_suffix(".parquet.tmp")
    with pq.ParquetWriter(tmp_steps, STEP_SCHEMA, compression="zstd") as writer:
        for k, path in enumerate(tqdm(io.raw_files(), desc="파싱")):
            table = pq.read_table(path, columns=RAW_COLUMNS)
            meta = table.drop_columns(["trajectory"]).to_pylist()
            trajs, steps = [], []
            for r, (row, messages) in enumerate(zip(meta, table.column("trajectory").to_pylist())):
                tid = io.traj_id(k, r)
                info, traj_steps = parse.parse_trajectory(messages)
                trajs.append({"traj_id": tid, "file_idx": k, "row_idx": r, **row, **info})
                for s in traj_steps:
                    head_counts[s["command_head"]] += 1
                    steps.append({"traj_id": tid, **s})
            traj_tables.append(pa.Table.from_pylist(trajs, schema=TRAJ_SCHEMA))
            writer.write_table(pa.Table.from_pylist(steps, schema=STEP_SCHEMA), row_group_size=50_000)
            del table, meta, trajs, steps
    pq.write_table(pa.concat_tables(traj_tables), io.TRAJECTORIES_PATH, compression="zstd")
    tmp_steps.replace(io.STEPS_PATH)

    total = sum(head_counts.values())
    counts = pl.DataFrame(
        [{"command_head": h, "count": c, "in_vocab": h in parse.COMMANDS} for h, c in head_counts.most_common()],
        schema={"command_head": pl.String, "count": pl.Int64, "in_vocab": pl.Boolean},
    ).with_columns(share=pl.col("count") / total)
    counts.head(200).write_csv(out_dir / "command_counts.csv")


def check(out_dir):
    traj = pl.read_parquet(io.TRAJECTORIES_PATH)
    steps = pl.scan_parquet(io.STEPS_PATH)
    s = {}

    s["trajectories"] = traj.height
    s["issues"] = traj["instance_id"].n_unique()
    s["traj_id_dup"] = traj.height - traj["traj_id"].n_unique()
    s["steps_total"] = int(traj["n_steps"].sum())
    s["steps_rows"] = steps.select(pl.len()).collect().item()
    s["success_rate"] = round(traj["target"].mean(), 4)
    s["models"] = {
        r["model_name"]: {"n": r["n"], "success_rate": round(r["sr"], 4)}
        for r in traj.group_by("model_name").agg(n=pl.len(), sr=pl.col("target").mean()).sort("n", descending=True).iter_rows(named=True)
    }
    s["median_steps"] = {
        "success": traj.filter(pl.col("target"))["n_steps"].median(),
        "failure": traj.filter(~pl.col("target"))["n_steps"].median(),
    }
    s["max_steps"] = int(traj["n_steps"].max())
    s["running_at_N"] = {}
    for n in (3, 5, 8, 10, 15, 20):
        alive = traj.filter(pl.col("n_steps") > n)
        s["running_at_N"][n] = {"n": alive.height, "success_rate": round(alive["target"].mean(), 4)}

    per_issue = traj.group_by("instance_id").agg(n=pl.len(), succ=pl.col("target").sum())
    s["traj_per_issue"] = {"median": per_issue["n"].median(), "max": int(per_issue["n"].max())}
    s["issues_never_success"] = per_issue.filter(pl.col("succ") == 0).height
    mixed = per_issue.filter((pl.col("succ") > 0) & (pl.col("succ") < pl.col("n")))
    s["issues_mixed"] = {"issues": mixed.height, "trajectories": int(mixed["n"].sum())}

    bad = traj.filter(~pl.col("role_pattern_ok"))
    s["role_pattern_bad"] = {"n": bad.height, "examples": bad["roles_head"].value_counts().sort("count", descending=True).head(5).to_dicts()}
    s["system_prompt_hashes"] = traj["system_hash"].value_counts().sort("count", descending=True).to_dicts()
    s["zero_step_trajectories"] = traj.filter(pl.col("n_steps") == 0).height

    st = steps.select(
        no_obs=(pl.col("obs_text").is_null()).sum(),
        no_command=(pl.col("command").is_null()).sum(),
        error_rate=pl.col("is_error").mean(),
        edit_failed_rate=pl.col("edit_failed").mean(),
        other_rate=(pl.col("command_name") == "other").mean(),
    ).collect().to_dicts()[0]
    s["steps_stats"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}

    # 고정 상한 K (fixed_cap_k.py와 같은 정의) — 스텝 테이블 기준으로 다시 계산
    n_steps, succ = traj["n_steps"].to_numpy().astype(int), traj["target"].to_numpy()
    s["fixed_cap"] = {}
    for K in (30, 35, 40, 50):
        lost = float((n_steps[succ] > K).mean())
        saved = float((n_steps - K).clip(0).sum() / n_steps.sum())
        s["fixed_cap"][K] = {"lost_success": round(lost, 4), "saved_steps": round(saved, 4)}

    (out_dir / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2))
    print(json.dumps(s, ensure_ascii=False, indent=2))

    # 제안서 수치와 대조
    assert s["trajectories"] == 80_036, s["trajectories"]
    assert s["issues"] == 3_591, s["issues"]
    assert s["traj_id_dup"] == 0
    assert s["steps_total"] == s["steps_rows"] == 2_115_623, (s["steps_total"], s["steps_rows"])
    assert [m["n"] for m in s["models"].values()] == [74_792, 4_053, 1_191], s["models"]
    assert round(s["success_rate"], 3) == 0.167, s["success_rate"]
    assert s["median_steps"] == {"success": 12, "failure": 19}, s["median_steps"]
    assert s["max_steps"] == 408, s["max_steps"]
    assert s["running_at_N"][10]["n"] == 59_708 and round(s["running_at_N"][10]["success_rate"], 3) == 0.138
    assert s["running_at_N"][20]["n"] == 34_059 and round(s["running_at_N"][20]["success_rate"], 3) == 0.064
    expected_cap = {30: (0.070, 0.294), 35: (0.049, 0.246), 40: (0.037, 0.208), 50: (0.022, 0.158)}
    for K, (lost, saved) in expected_cap.items():
        got = s["fixed_cap"][K]
        assert (round(got["lost_success"], 3), round(got["saved_steps"], 3)) == (lost, saved), (K, got)
    print("\n완료 기준 통과")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="결과 파일이 있어도 다시 파싱")
    args = ap.parse_args()

    out_dir = io.RESULTS_DIR / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.force or not (io.TRAJECTORIES_PATH.exists() and io.STEPS_PATH.exists()):
        build(out_dir)
    else:
        print("결과 파일이 있어 파싱을 건너뜀 (--force로 다시 만듦)")
    check(out_dir)


if __name__ == "__main__":
    main()
