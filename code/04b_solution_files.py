"""Phase 5.5 사후 분석 (피처 아님): 정답 파일을 열었는지와 성공의 관계

실행: uv run --no-sync python code/04b_solution_files.py

정답 패치는 데이터에 없어서 근사를 쓴다 (afp/leakage.py 5.5 절): 같은 이슈의 다른 성공 궤적 패치가 고친 기존 비테스트 파일.
원본의 결과 전용 컬럼(패치)은 afp/leakage.py만 읽는다. 이 스크립트의 출력은 results/phase5/의 표뿐이고
data/processed/의 피처·예측 입력에는 들어가지 않는다 (M1 학습과 무관, 04_m1.py보다 나중에 돌려도 된다).

출력: results/phase5/solution_file_open.csv (범위 × 성공 여부), solution_file_open_by_n.csv (N별, 앞 N스텝 안에서 열었는지),
      solution_file_hint_check.csv (본문에 고칠 파일 목록이 있는 13개 이슈: 근사 vs 목록), solution_files_summary.json
"""
import polars as pl

from afp import evaluate, io, leakage

FLOAT = 6
DATA_CARD = {"success": 0.83, "failure": 0.40}  # 제안서 §5가 인용한 데이터 카드 값 (정답 파일을 한 번이라도 연 비율)


def rate_rows(t: pl.DataFrame, scope: str, n=None) -> list[dict]:
    rows = []
    for succ in (True, False):
        g = t.filter(pl.col("target") == succ)
        rows.append({"scope": scope, "N": n, "target": "success" if succ else "failure", "n_traj": g.height,
                     "n_issues": g["instance_id"].n_unique(), "opened_rate": float(g["opened"].mean()) if g.height else None})
    return rows


def main():
    out_dir = io.RESULTS_DIR / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj = io.load_trajectories(["traj_id", "file_idx", "row_idx", "instance_id", "model_name", "target", "n_steps", "issue_text"])
    files = leakage.load_patch_files(traj)
    proxy = leakage.solution_proxy(traj, files)
    steps = io.scan_steps(["traj_id", "step_idx", "open_file"]).filter(pl.col("open_file").is_not_null()).collect()
    first = leakage.opened_files(steps, proxy, "proxy")
    t = (traj.join(proxy.select("traj_id"), on="traj_id", how="semi").join(first, on="traj_id", how="left")
         .with_columns(opened=pl.col("first_step").is_not_null()))
    ids70 = pl.concat([s.select("traj_id") for s in evaluate.load_eval_sets().values()])
    t70 = t.join(ids70, on="traj_id", how="semi")

    rows = rate_rows(t, "all_models") + rate_rows(t70, "70b_eval_sets")
    pl.DataFrame(rows).write_csv(out_dir / "solution_file_open.csv", float_precision=FLOAT)
    by_n = []
    for n in evaluate.N_GRID:
        r = t70.filter(pl.col("n_steps") > n).with_columns(opened=pl.col("first_step").is_not_null() & (pl.col("first_step") <= n))
        by_n += rate_rows(r, "70b_eval_sets_first_n_steps", n)
    pl.DataFrame(by_n).write_csv(out_dir / "solution_file_open_by_n.csv", float_precision=FLOAT)

    # 근사 대조: 본문 끝에 "고칠 파일 목록"이 있는 이슈 (정답 파일을 직접 아는 유일한 경우)
    from afp import baselines
    hint = (traj.select("instance_id", "issue_text").filter(pl.col("issue_text").str.contains(baselines.HINT, literal=True))
            .unique("instance_id").with_columns(hint=pl.col("issue_text").map_elements(leakage.hint_files, return_dtype=pl.List(pl.String)))
            .filter(pl.col("hint").list.len() > 0).select("instance_id", "hint").sort("instance_id"))
    issue_proxy = (traj.select("traj_id", "instance_id", "target").join(files, on="traj_id").filter(pl.col("target"))
                   .explode("files").drop_nulls("files").group_by("instance_id").agg(proxy=pl.col("files").unique().sort()))
    hc = hint.join(issue_proxy, on="instance_id", how="left").with_columns(
        has_proxy=pl.col("proxy").is_not_null(),
        hint_in_proxy=pl.col("hint").list.set_intersection(pl.col("proxy").fill_null([])).list.len(),
        n_hint=pl.col("hint").list.len(), n_proxy=pl.col("proxy").list.len().fill_null(0))
    hc.with_columns(pl.col("hint").list.join(";"), pl.col("proxy").list.join(";")).write_csv(out_dir / "solution_file_hint_check.csv")

    excluded = traj.join(proxy.select("traj_id"), on="traj_id", how="anti")
    s = {
        "definition": "정답 파일 근사 = 같은 이슈의 다른 성공 궤적 generated_patch가 고친 기존 비테스트 파일 (자기 패치 제외). "
                      "열람 = 관측 상태 줄의 열린 파일 경로가 '/근사 경로'로 끝남",
        "data_card": DATA_CARD,
        "opened_rate": {f"{r['scope']}/{r['target']}": r["opened_rate"] for r in rows},
        "n_traj_with_proxy": {"all_models": t.height, "70b_eval_sets": t70.height},
        "n_traj_without_proxy": {"all_models": excluded.height, "success": int(excluded["target"].sum()),
                                 "issues": excluded["instance_id"].n_unique()},
        "by_n_70b": [{k: r[k] for k in ("N", "target", "n_traj", "opened_rate")} for r in by_n],
        "hint_check": {"issues": hc.height, "with_proxy": int(hc["has_proxy"].sum()),
                       "hint_files": int(hc["n_hint"].sum()), "hint_files_in_proxy": int(hc["hint_in_proxy"].sum())},
    }
    io.write_summary(out_dir / "solution_files_summary.json", s)
    with pl.Config(tbl_rows=40, float_precision=3):
        print(pl.DataFrame(rows))
        print(pl.DataFrame(by_n))
        print(hc.select("instance_id", "n_hint", "n_proxy", "hint_in_proxy"))
    print(s["n_traj_without_proxy"], s["hint_check"])


if __name__ == "__main__":
    main()
