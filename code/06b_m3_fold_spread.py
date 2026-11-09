"""Phase 7.4 사후 분석: fold 모델 하나의 test AUROC와 5개 평균 점수의 test AUROC

실행: uv run --no-sync python code/06b_m3_fold_spread.py (06_m3.py --final 뒤)

판별 지표는 fold 평균 점수로 내고(P4-D2) 운영점은 fold 모델별로 채점한다(P4-D6). 평균하면 AUROC가 얼마나 오르는지를
모델마다 비교한다. 결과를 본 뒤의 설명용이라 아무것도 고르지 않는다. 예측 파일은 모두 이 머신의 것이다 (P6-D10).

출력: results/phase7/fold_spread.csv
"""
import polars as pl

from afp import evaluate, io, preds

MODELS = ("m1_xgb", "m1_xgb_b1", "m2_lstm", "m2_lstm_b1", "m2x_nocum", "m2x_nocum_b1", "m3_bert", "m3_bert_b1")
N_GRID = (5, 10, 20)


def auroc(trajs: pl.DataFrame, p: pl.DataFrame, n: int) -> float:
    return evaluate.discrimination_row(trajs, evaluate.split_preds(p, trajs, n), n)["auroc"]


def main():
    sets = evaluate.load_eval_sets()
    rows = []
    for m in MODELS:
        mean, folds = preds.read(m), preds.read_test_folds(m)
        k_all = sorted(folds["fold"].unique().to_list())
        for n in N_GRID:
            per = [auroc(sets["test"], folds.filter(pl.col("fold") == k).drop("fold"), n) for k in k_all]
            ens = auroc(sets["test"], mean, n)
            single = sum(per) / len(per)
            rows.append({"model": m, "N": n, "train_oof_auroc": auroc(sets["train"], mean, n), "test_auroc_mean_score": ens,
                         "test_auroc_fold_mean": single, "test_auroc_fold_min": min(per), "test_auroc_fold_max": max(per),
                         "ensemble_gain": ens - single})
    out = pl.DataFrame(rows).with_columns(pl.selectors.float().round(io.SUMMARY_DIGITS))
    path = io.RESULTS_DIR / "phase7" / "fold_spread.csv"
    out.write_csv(path)
    print(out)
    print(f"→ {path}")


if __name__ == "__main__":
    main()
