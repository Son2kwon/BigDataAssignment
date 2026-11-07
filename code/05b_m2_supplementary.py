"""Phase 6.5 보조 실험 (PLAN 11절 P6-D9, docs/phase6.md 6.5 사전 등록): 누적 집계 없는 M2의 test 평가. 주 결과는 바꾸지 않는다.

실행:
  uv run --no-sync python code/05b_m2_supplementary.py                  # 개발: B1 포함판 학습 + train OOF만 출력 (test 안 봄, 결과 파일 안 씀)
  uv run --no-sync python code/05b_m2_supplementary.py --final          # 최종: test 한 번 채점, results/phase6/supplementary/
  uv run --no-sync python code/05b_m2_supplementary.py --smoke [--final] # 점검: 05_m2.py --smoke 캐시로 전체 경로 (실제 test 안 씀)

모델 (사전 등록, 새 선택 없음):
- m2x_nocum    = m2_lstm의 고른 설정에서 누적 집계만 뺀 것. 개발 실행(05_m2.py)의 절제 모델 그대로 (재학습 없음)
- m2x_nocum_b1 = m2_lstm_b1의 고른 설정에서 누적 집계만 뺀 것 (같은 규칙으로 새로 학습. train만 쓴다)
평가: 05_m2.py 최종과 같은 함수 (운영점은 train OOF에서, test는 fold 모델별 채점 평균 P4-D6, B1 test 입력은 fold k P4-D7 ①).
짝지은 비교 (같은 재표본): 누적 집계의 몫 (m2x_nocum − m2_lstm, m2x_nocum_b1 − m2_lstm_b1), M1 대비 (− m1_xgb, − m1_xgb_b1).

읽기만 하는 것: results/phase4·5·6/ (주 결과), data/processed/preds/ (주 M2·M1·B1 예측). 쓰는 곳: results/phase6/supplementary/,
data/processed/preds/m2x_*.parquet, data/processed/m2/models/m2x_nocum_b1/ (git 밖).
"""
import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from afp import evaluate, io, lstm, metrics, preds, sequences, split

m2 = importlib.import_module("05_m2")  # 주 실험의 학습·채점·평가 경로를 그대로 쓴다

SUPP = {"m2x_nocum": "m2_lstm", "m2x_nocum_b1": "m2_lstm_b1"}  # 보조 모델 → 설정을 가져오는 주 모델 (누적 집계만 뺀다)
PAIRS = [("m2x_nocum", "m2_lstm"), ("m2x_nocum_b1", "m2_lstm_b1"),  # 누적 집계를 뺀 효과 (test)
         ("m2x_nocum", "m1_xgb"), ("m2x_nocum_b1", "m1_xgb_b1")]  # 같은 B1 조건의 가장 강한 M1 대비 (주 실험과 같은 짝)
FLOAT = m2.FLOAT


def supp_cfgs(dev_summary: dict) -> dict[str, lstm.M2Config]:
    """보조 모델 설정 = 주 모델의 고른 설정에서 use_cum만 끈다 (05_m2.py 절제와 같은 규칙)."""
    out = {s: lstm.copy_config(lstm.M2Config(**dev_summary["selected"][base]["cfg"]), use_cum=False) for s, base in SUPP.items()}
    if m2.cfg_key(out["m2x_nocum"]) != dev_summary["ablation"]["m2x_nocum"]["config"]:
        raise AssertionError("m2x_nocum 설정이 개발 실행의 절제 모델과 다르다")
    return out


def train_oof(args, paths, sets, costs, data_tr, b1_tr, b1_file, device, cfgs) -> dict:
    """fold 모델 학습(저장돼 있으면 읽기만) + train OOF 점수·운영점·좋은 N. test는 보지 않는다."""
    log_rows = m2.run_training(list(cfgs.items()), paths, data_tr, b1_file, device, args.jobs, args.force)
    out = {"log": log_rows}
    for s, c in cfgs.items():
        r = m2.collect(s, c, paths, data_tr, b1_tr, device)
        assert r["seen_issue_scores"] == 0
        sc = m2.oof_scores(s, r["train"], sets["train"])
        ops = m2.train_ops(sets["train"], r["train"], costs["train"])
        out[s] = {"cfg": c, "train": r["train"], "scores": sc, "ops": ops, "good_n": m2.good_n_rows(s, sc, ops)}
    return out


def dev_report(res, paths, dev_summary):
    """개발 출력: 보조 모델과 주 M2·M1의 train OOF AUROC, train 운영점, 좋은 N, epoch 상한."""
    rows = [{"model": s, "config": m2.cfg_key(v["cfg"]), "N": n, "train_oof_auroc": v["scores"][f"oof_auroc_N{n}"]}
            for s, v in res.items() if s in SUPP for n in evaluate.N_GRID]
    abl = pl.read_csv(paths.out / "ablation_oof.csv").filter(pl.col("model").is_in(["m2_lstm", "m2_lstm_b1", "m1_xgb", "m1_xgb_b1"]))
    tbl = pl.concat([pl.DataFrame(rows), abl.select("model", "config", pl.col("N").cast(pl.Int64), "train_oof_auroc")])
    ops = pl.DataFrame([{"model": s, **r} for s in SUPP for r in res[s]["ops"]])
    hit = pl.DataFrame(res["log"]).filter(pl.col("hit_max_epochs")).height
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=220, float_precision=4):
        print(tbl.pivot(on="N", index=["model", "config"], values="train_oof_auroc"))
        print(ops.filter(pl.col("N").is_in([10, 15, 20])).sort("N", "policy", "model"))
        print(pl.DataFrame([g for s in SUPP for g in res[s]["good_n"]]).filter(pl.col("selected_auroc_first") | pl.col("selected_saved_first")))
    d = res["m2x_nocum"]["scores"]["oof_auroc_mean"] - dev_summary["ablation"]["m2x_nocum"]["oof_auroc_mean"]
    print(f"m2x_nocum OOF AUROC 평균이 개발 실행 절제 값과 같은지: 차이 {d:.2e}")
    if hit:
        print(f"\n[주의] 최고 epoch가 max_epochs인 망 {hit}개. 보고 후 --final")
    else:
        print("\n개발 실행 완료 (train OOF만 봄): epoch 상한 문제 없음. --final 을 실행해도 된다")
    return hit


def with_ci(disc: pl.DataFrame, ci: pl.DataFrame) -> pl.DataFrame:
    """판별 표 + 부트스트랩 구간 (discrimination_compare.csv와 같은 컬럼)."""
    w = ci.select("model", "split", "N", "stat", "lo", "hi").pivot(on="stat", index=["model", "split", "N"], values=["lo", "hi"])
    w = w.rename({c: (c.split("_", 1)[1] + "_" + c.split("_", 1)[0]) for c in w.columns if c.startswith(("lo_", "hi_"))})
    d = disc.join(w, on=["model", "split", "N"], how="left")
    keep = ["model", "split", "N", "n_running", "base_rate", "within_n_issues"]
    for c in evaluate.PAIR_DISC:
        keep += [c] + [x for x in (f"{c}_lo", f"{c}_hi") if x in d.columns]
    return d.select(keep)


def final(args, paths, sets, costs, data_tr, b1_tr, b1_file, frames, device, dev_summary, cfgs, lap):
    res = train_oof(args, paths, sets, costs, data_tr, b1_tr, b1_file, device, cfgs)
    lap("fold 모델 (저장본) + train OOF")
    data_te = sequences.load(sets["test"])
    b1_te = [sequences.b1_matrix(data_te, f) for f in frames["test"]]
    written, written_folds, model_preds = {}, {}, {}
    for s, c in cfgs.items():
        r = m2.collect(s, c, paths, data_tr, b1_tr, device, data_te, b1_te)
        assert r["seen_issue_scores"] == 0 and r["train"].equals(res[s]["train"])
        config = {"learner": "lstm", "cfg": c.as_dict(), "derived_from": f"{SUPP[s]} 고른 설정에서 use_cum=False (P6-D9 사전 등록)",
                  "step_features": sequences.STEP_FEATURES, "cum_features": [], "static": sequences.STATIC_FEATURES,
                  "b1_input": f"train={m2.B1_MODEL} OOF, test fold k={m2.B1_MODEL} fold k (P4-D7 ①)" if c.use_b1 else None, "device": device}
        written[s] = preds.write(s, pl.concat([r["train"], r["test_mean"]]), sets, evaluate.N_GRID, config, "5fold-oof/test-fold-mean",
                                 force=args.force)
        written_folds[s] = preds.write_test_folds(s, r["test_folds"], sets["test"], evaluate.N_GRID, split.N_FOLDS, force=args.force)
        model_preds[s] = (preds.read(s), preds.read_test_folds(s))
    lap("test 채점·예측 저장")

    # 비교 대상: 주 M2 (+ 점검이 아니면 M1). 같은 평가 함수로 운영점을 다시 고른다 (주 결과 파일과 같아야 한다)
    others = list(m2.MODEL_SPECS) + ([] if args.smoke else list(m2.M1_PAIR.values()))
    missing = [b for b in others if not preds.path(b).exists()]
    if missing:
        sys.exit(f"예측 파일이 없다: {missing}. 05_m2.py --final{' --smoke' if args.smoke else ''} / 04_m1.py 를 먼저 실행한다.")
    other_preds = {b: (preds.read(b), preds.read_test_folds(b)) for b in others}
    pairs = [(a, b) for a, b in PAIRS if b in other_preds]

    out = paths.out / "supplementary"
    out.mkdir(parents=True, exist_ok=True)
    disc = pl.concat([evaluate.discrimination_table(s, model_preds[s][0], sets, evaluate.N_GRID) for s in SUPP])
    disc.write_csv(out / "discrimination.csv", float_precision=FLOAT)
    cap_train = metrics.fixed_cap_curve(sets["train"], costs["train"], range(1, 81))
    k_star = metrics.select_cap(cap_train, metrics.SELECT_MAX_LOSS)["K"]
    main_rows, sens_rows, chosen = evaluate.operating_points(model_preds, sets, costs, k_star)
    _, _, chosen_o = evaluate.operating_points(other_preds, sets, costs, k_star)
    chosen.update(chosen_o)
    main_ops = pl.read_csv(paths.out / "operating_points.csv", schema_overrides=evaluate.OP_SCHEMA).filter(
        (pl.col("split") == "train") & pl.col("model").is_in(list(m2.MODEL_SPECS)))
    # 결과 CSV는 소수 FLOAT자리라 τ는 그 반올림 폭 안에서 같으면 같다
    main_same = main_ops.height == len(m2.MODEL_SPECS) * len(evaluate.N_GRID) * len(evaluate.POLICIES) and all(
        abs(chosen[(r["model"], r["N"], r["policy"])][0] - r["tau"]) <= 10 ** -FLOAT and chosen[(r["model"], r["N"], r["policy"])][1] == r["cap"]
        for r in main_ops.iter_rows(named=True))
    cap_main, cap_sens = evaluate.cap_rows(sets, costs, k_star, cap_train)
    lap("판별 지표·운영점 선택·적용")

    tasks_b, keys = evaluate.bootstrap_tasks(model_preds, sets, costs, chosen, k_star, {s: evaluate.N_GRID for s in SUPP}, args.n_boot)
    cap_tasks, cap_keys = evaluate.cap_bootstrap_tasks(sets, costs, k_star, args.n_boot)
    ptasks, pkeys = m2.paired_tasks(pairs, {**model_preds, **other_preds}, sets, costs, chosen, args.n_boot)
    r_all = evaluate.parallel(evaluate.boot_task, tasks_b + cap_tasks + ptasks, args.cpu_jobs)
    nb = len(keys) + len(cap_keys)
    boots = dict(zip(keys + cap_keys, r_all[:nb]))
    pair_boots = dict(zip(pkeys, r_all[nb:]))
    ci = evaluate.discrimination_ci(boots)
    ci.write_csv(out / "discrimination_ci.csv", float_precision=FLOAT)
    ops_df = evaluate.operating_table(cap_main + main_rows, boots)
    ops_df.write_csv(out / "operating_points.csv", float_precision=FLOAT)
    evaluate.sensitivity_table(cap_sens + sens_rows).write_csv(out / "operating_points_sensitivity.csv", float_precision=FLOAT)
    pair_df = pl.concat([b.with_columns(a=pl.lit(a), b=pl.lit(bm), N=pl.lit(n)) for (a, bm, n), b in pair_boots.items()]) \
        .select("a", "b", "N", "stat", "estimate", "lo", "hi", "n_boot")
    pair_df.write_csv(out / "paired.csv", float_precision=FLOAT)
    lap(f"부트스트랩 ({len(r_all)}개 × {args.n_boot}회)")

    sens2 = pl.DataFrame(m2.b1_input_sensitivity("m2x_nocum_b1", cfgs["m2x_nocum_b1"], paths, data_te, frames, sets, costs, chosen,
                                                 model_preds, device)).with_columns(diff=pl.col("sens_2") - pl.col("main_1"))
    sens2.write_csv(out / "b1_input_sensitivity.csv", float_precision=FLOAT)
    lap("② 민감도")

    # 주 결과·M1과 나란히 (주 결과 파일 그대로 읽는다)
    dmain = pl.read_csv(paths.out / "discrimination_compare.csv", schema_overrides={"N": pl.Int64})
    dsupp = with_ci(disc, ci)
    dcomp = pl.concat([dmain.select(dsupp.columns).cast(dsupp.schema), dsupp]).sort("split", "N", "model")
    dcomp.write_csv(out / "discrimination_compare.csv", float_precision=FLOAT)
    omain = pl.read_csv(paths.out / "operating_points_compare.csv", schema=ops_df.schema)
    ocomp = pl.concat([omain, ops_df.filter(pl.col("model") != "b0_cap")])
    nsum = evaluate.n_summary(ocomp.filter(pl.col("model").is_in(["b0_cap", *m2.M1_MODELS, *m2.MODEL_SPECS, *SUPP])))
    nsum.write_csv(out / "n_summary.csv", float_precision=FLOAT)
    pl.DataFrame([g for s in SUPP for g in res[s]["good_n"]]).write_csv(out / "good_n.csv", float_precision=FLOAT)
    pl.DataFrame(res["log"]).sort("model", "config", "fold", "seed").write_csv(out / "train_log.csv", float_precision=FLOAT)
    lap("비교 표")

    test_ops = ops_df.filter((pl.col("split") == "test") & pl.col("model").is_in(list(SUPP)))
    keep = ["d_auroc", "d_within_pair", "combined.d_saved_steps", "combined.d_saved_tokens", "single.d_saved_steps",
            "single.d_saved_tokens", "combined.d_lost_success", "single.d_lost_success"]
    pair_sum = pair_df.filter(pl.col("stat").is_in(keep)).sort("a", "b", "stat", "N")
    d_abl = res["m2x_nocum"]["scores"]["oof_auroc_mean"] - dev_summary["ablation"]["m2x_nocum"]["oof_auroc_mean"]
    hit = pl.DataFrame(res["log"]).filter(pl.col("hit_max_epochs")).height
    s = {
        "pre_registration": "docs/phase6.md 6.5 (PLAN 11절 P6-D9). 주 결과는 바꾸지 않는다",
        "models": {k: {"config": m2.cfg_key(c), "cfg": c.as_dict(), "derived_from": SUPP[k]} for k, c in cfgs.items()},
        "k_star": k_star,
        "preds": {k: {"rows": w["rows"], "written": w["written"]} for k, w in written.items()},
        "train_oof_auroc_mean": {k: res[k]["scores"]["oof_auroc_mean"] for k in SUPP},
        "discrimination_N10": disc.filter(pl.col("N") == 10).select("model", "split", "auroc", "pr_auc", "within_pair").to_dicts(),
        "n_summary_supp": nsum.filter(pl.col("model").is_in(list(SUPP))).to_dicts(),
        "judged_and_saves_more_than_k_star": evaluate.judged_and_saves_more(ops_df).to_dicts(),
        "paired": pair_sum.to_dicts(),
        "b1_input_sensitivity_max_abs_diff": sens2.with_columns(a=pl.col("diff").abs()).group_by("kind", "metric").agg(
            pl.col("a").max()).sort("kind", "metric").to_dicts(),
        "fold_models_hit_max_epochs": hit,
        "bootstrap": {"n_boot": args.n_boot, "seed": io.SEED},
        "smoke": args.smoke,
    }
    crit = {
        "S1_preds_cover_running_trajectories": all(w["coverage"] for w in written.values()),
        "S2_no_seen_issue": True,  # collect에서 assert
        "S3_nocum_oof_equals_dev_ablation": abs(d_abl) <= 1e-9,
        "S4_k_star": args.smoke or k_star == 40,
        "S5_fold_files_mean_matches": all(w["max_abs_mean_diff"] is not None and w["max_abs_mean_diff"] <= preds.FOLD_MEAN_TOL
                                          for w in written_folds.values()),
        "S6_operating_points": test_ops.height == len(SUPP) * len(evaluate.N_GRID) * len(evaluate.POLICIES)
        and test_ops["lost_success_hi"].null_count() == 0 and test_ops["meets_judge_ci_hi"].null_count() == 0,
        "S7_main_m2_operating_points_reproduced": main_same,
        "S8_b1_sensitivity": sens2.height > 0 and sens2["sens_2"].null_count() == 0,
        "S9_paired": pair_df.height > 0 and pair_df["estimate"].null_count() == 0 and len(pairs) == (2 if args.smoke else 4),
    }
    s["criteria"] = crit
    s["criteria_note"] = "pytest 전체 통과, 재실행 시 results/·preds/ 변화 없음, 주 결과(results/phase6/ 위 단계) 불변은 스크립트 밖에서 확인한다"
    io.write_summary(out / "summary.json", s)
    with pl.Config(tbl_rows=80, tbl_cols=20, tbl_width_chars=240, float_precision=4):
        print(dcomp.filter((pl.col("split") == "test") & pl.col("model").is_in([*SUPP, *m2.MODEL_SPECS, "m1_xgb", "m1_xgb_b1"])).select(
            "model", "split", "N", "auroc", "auroc_lo", "auroc_hi", "within_pair").sort("split", "N", "model"))
        print(nsum.filter(pl.col("model").is_in(["b0_cap", *SUPP, *m2.MODEL_SPECS, "m1_xgb", "m1_xgb_b1"])))
        print(pair_sum)
    print(json.dumps({"criteria": crit}, ensure_ascii=False, indent=2))
    failed = [k for k, v in crit.items() if not v]
    assert not failed, failed
    print("\n완료 기준 통과")


def main():
    ap = argparse.ArgumentParser(description="Phase 6.5 보조 실험: 누적 집계 없는 M2")
    ap.add_argument("--final", action="store_true", help="test를 한 번 채점하고 results/phase6/supplementary/에 쓴다 (없으면 train OOF만 출력)")
    ap.add_argument("--smoke", action="store_true", help="05_m2.py --smoke 캐시로 경로 점검 (실제 test 안 씀)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--cpu-jobs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--n-boot", type=int, default=evaluate.N_BOOT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = lstm.resolve_device(args.device)
    if args.jobs is None:
        args.jobs = 4 if device == "cuda" else min(16, os.cpu_count() or 1)
    paths = m2.Paths(args.smoke)
    dev_path = paths.out / "summary_dev.json"
    if not dev_path.exists() or not (paths.out / "operating_points_compare.csv").exists():
        sys.exit(f"{paths.out}에 주 실험 결과가 없다. 먼저 05_m2.py{' --smoke' if args.smoke else ''} 개발·--final 을 실행한다.")
    dev_summary = json.loads(dev_path.read_text())
    if dev_summary["device"] != device:
        sys.exit(f"주 실험은 {dev_summary['device']}로 학습했다. 같은 --device로 실행한다.")
    frames = m2.load_b1_frames(args.final, args.smoke)  # 실제 예측 파일에서 읽는다 (점검 모드의 경로 변경 전)
    if args.smoke:
        preds.PREDS_DIR = paths.preds
        args.n_boot = min(args.n_boot, 20)
    t0 = time.time()
    timings = {}

    def lap(name):
        timings[name] = round(time.time() - t0 - sum(timings.values()), 1)
        print(f"[{timings[name]:>6.1f}s] {name}", flush=True)

    print(f"모드: 보조 {'최종' if args.final else '개발'}{' (점검)' if args.smoke else ''}, 장치 {device}", flush=True)
    sets = evaluate.load_eval_sets()
    if args.smoke:
        sets = m2.smoke_sets(sets)
    costs = evaluate.load_costs(sets)
    data_tr = sequences.load(sets["train"])
    if data_tr.fingerprint != dev_summary["data"]["fingerprint"]:
        sys.exit("train 시퀀스가 주 실험 개발 실행 때와 다르다.")
    seq_meta = paths.seq_train / "meta.txt"
    if not seq_meta.exists() or seq_meta.read_text().split()[0] != data_tr.fingerprint:
        sequences.save_arrays(data_tr, paths.seq_train)
    b1_tr = sequences.b1_matrix(data_tr, frames["train"])
    b1_file = Path(paths.cache) / "b1_train.npy"
    if not b1_file.exists() or not np.array_equal(np.load(b1_file), b1_tr, equal_nan=True):
        np.save(b1_file, b1_tr)
    cfgs = supp_cfgs(dev_summary)
    print({k: m2.cfg_key(c) for k, c in cfgs.items()}, flush=True)
    lap(f"train 시퀀스 ({data_tr.n}궤적)")
    if not args.final:
        res = train_oof(args, paths, sets, costs, data_tr, b1_tr, b1_file, device, cfgs)
        lap("학습 + train OOF")
        dev_report(res, paths, dev_summary)
        print(json.dumps({"timings": timings}, ensure_ascii=False))
        return
    final(args, paths, sets, costs, data_tr, b1_tr, b1_file, frames, device, dev_summary, cfgs, lap)
    print(json.dumps({"timings": timings}, ensure_ascii=False))


if __name__ == "__main__":
    main()
