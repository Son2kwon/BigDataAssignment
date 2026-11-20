# Final analysis of early failure prediction

Final run `120c121ea083ab4d` from commit `067a40d04ee5`, executed on an NVIDIA GeForce RTX 5070. The analysis uses the single published run. It describes 70B SWE-agent evaluation, supplementary model studies, and the 8B/405B transfer checks.

## Findings

- **K* is a competitive baseline.** At K=40, test success loss was 2.02 [1.30, 2.90]%, with 19.83% of steps saved and 34.43% of estimated context tokens saved. The proposal cap of 35 saves more, with a test loss upper bound of 4.13%.

- `m1_xgb` combined policy met the loss criterion at 8/8 prefixes. It also showed positive paired savings against K* in steps at N=3, 8, 10, 15, 20, 25, 30 and in tokens at N=10, 15, 20, 25, 30.

- `m1_xgb_b1` combined policy met the loss criterion at 8/8 prefixes. It also showed positive paired savings against K* in steps at N=3, 10, 15, 20, 25, 30 and in tokens at N=3, 15, 20, 25, 30.

- `m1_xgb_s4` combined policy met the loss criterion at 6/6 prefixes. It also showed positive paired savings against K* in steps at N=3, 5, 8, 10, 15, 20 and in tokens at N=15, 20.

- `m1_xgb_b1_s4` combined policy met the loss criterion at 3/6 prefixes. It also showed positive paired savings against K* in steps at N=3, 15, 20 and in tokens at N=3, 15, 20.

- `m2x_nocum` combined policy met the loss criterion at 6/6 prefixes. It also showed positive paired savings against K* in steps at N=3, 5, 8, 10, 15, 20 and in tokens at N=3, 10, 15, 20.

- `m3_bert_b1` combined policy met the loss criterion at 3/3 prefixes. It also showed positive paired savings against K* in steps at N=20 and in tokens at N=20.

- **Adding B1 changed overall and within-issue discrimination differently.** Across 31 matched model-and-prefix cells, overall AUROC rose in 25. Within-issue AUROC rose in 17. These are descriptive counts across related comparisons. The paired tables show uncertainty where the run estimated it.

- Cross-model transfer has smaller effective sample sizes and different failure pressures. Its results use the same 70B-trained policies and fixed-cap values, applied within each transfer population.

[Operating points](tables/operating_points.csv) · [Loss and savings across every N](plots/diagnostics/all_models_outcome_matrix.png) · [Held-out discrimination by family](plots/all_models/discrimination_test.png)

## Data, comparability and the decision rule

The parser verified 12 parquet shards with 80,036 trajectories. The 70B split has 62,311 training trajectories from 2,989 issues and 12,481 test trajectories from 602 issues. There were 0 shared issues and 0 training issues spanning multiple folds. The five training folds use StratifiedGroupKFold.

For a prefix of length N, discrimination is computed among trajectories still running after N. Overall AUROC and PR-AUC include across-issue difficulty. The primary within-issue AUROC pools success–failure pairs from the same issue, so issues with more such pairs carry more weight. The issue-average AUROC weights those issues equally. Later prefixes have a different survivor population, so increases in a curve across N reflect both added evidence and changes in who remains. Model contrasts match N and population.

The single policy evaluates one failure-score threshold at N. The combined policy adds a hard cap to that check. Selection maximized step savings subject to at most 3.5% successful training runs lost, using only train out-of-fold scores. Test judgment requires the upper 95% interval bound on successful runs lost to be at most 5%. Test discrimination uses the mean of five fold scores. For test stopping, the policy is applied separately to each fold model and the resulting metrics are averaged. Every B1 input on the primary test path is matched to its fold.

Intervals use 1,000 issue-level bootstrap draws, with paired resamples for differences. They describe sampling uncertainty conditional on the fitted models, selected hyperparameters and fixed policy. Read them as pointwise intervals across prefixes. The success-loss denominator is all successful runs in the split. The cost denominators include all runs in the split, including runs finished before N. Context-token savings use the accumulated character length of each step's input context as a proxy for previously seen input tokens. Predictor inference time, KV caching, and other billing details were outside that cost measure.

Predictors use observable prefix information. Terminal outcome metadata and correct-patch information are confined to verification or retrospective diagnostics. The project uses binary success labels to train the predictors and uses fit-fold outcomes to construct the special cost weights. Held-out labels are reserved for evaluation.

[Parse audit](phases/phase1/summary.json) · [Split and leakage audit](phases/phase2/summary.json) · [Survivor population and base rate](plots/diagnostics/population_by_prefix.png) · [Registered evaluation rules](../../PLAN.md)

## Matched comparison at N=10

N=10 is a common reporting point across the scored families. The operating-point columns use the combined policy. Loss and absolute savings are percentages. Differences from K* are in percentage points. Brackets give 95% intervals. The all-prefix summary below carries both policies.

| Model | AUROC | PR-AUC | Within-issue | Success lost % [95%] | Steps saved % [95%] | Δ steps vs K* pp [95%] | Tokens saved % [95%] | Δ tokens vs K* pp [95%] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K=40 fixed cap K* | — | — | — | 2.02 [1.30, 2.90] | 19.83 [18.02, 21.70] | — | 34.43 [32.05, 36.85] | — |
| K=35 proposal | — | — | — | 3.07 [2.13, 4.13] | 23.46 [21.55, 25.43] | +3.63 [+3.46, +3.81] | 40.68 [38.32, 43.04] | +6.26 [+5.95, +6.57] |
| `b0_err` | 0.5284 | 0.1431 | 0.5509 | 2.02 [1.30, 2.90] | 20.21 [18.33, 22.22] | +0.39 [+0.24, +0.57] | 34.83 [32.38, 37.31] | +0.40 [+0.24, +0.58] |
| `b0_editfail` | 0.5270 | 0.1438 | 0.5338 | 2.02 [1.30, 2.90] | 20.02 [18.18, 21.89] | +0.20 [+0.09, +0.34] | 34.69 [32.25, 37.17] | +0.26 [+0.13, +0.43] |
| `b1_static` | 0.6068 | 0.1849 | 0.5000 | 3.49 [1.54, 6.11] | 20.83 [18.28, 23.79] | +1.00 [-0.44, +2.67] | 33.26 [30.09, 36.61] | -1.17 [-2.77, +0.70] |
| `b1_n` | 0.6003 | 0.1737 | 0.5000 | 3.05 [1.14, 5.56] | 20.95 [18.42, 23.71] | +1.12 [-0.24, +2.63] | 33.57 [30.39, 36.88] | -0.86 [-2.35, +0.83] |
| `ref_oracle_loo` | 0.9377 | 0.6876 | — | — | — | — | — | — |
| `m1_lr` | 0.6805 | 0.2884 | 0.6117 | 2.59 [1.77, 3.63] | 20.94 [18.83, 23.20] | +1.12 [+0.64, +1.63] | 34.40 [31.85, 37.06] | -0.03 [-0.44, +0.42] |
| `m1_lr_b1` | 0.6948 | 0.2803 | 0.6125 | 3.55 [2.34, 4.99] | 21.95 [19.67, 24.28] | +2.12 [+1.25, +3.00] | 33.91 [31.16, 36.62] | -0.51 [-1.57, +0.57] |
| `m1_xgb` | 0.7014 | 0.3066 | 0.6111 | 2.24 [1.49, 3.13] | 21.04 [19.05, 23.21] | +1.22 [+0.86, +1.59] | 35.31 [32.82, 37.82] | +0.89 [+0.47, +1.33] |
| `m1_xgb_b1` | 0.7140 | 0.2969 | 0.6179 | 3.18 [1.86, 4.87] | 22.41 [19.83, 25.11] | +2.59 [+1.31, +4.06] | 33.50 [30.36, 36.80] | -0.93 [-2.52, +0.79] |
| `m1w_xgb` | 0.6978 | 0.3033 | 0.6140 | 2.23 [1.51, 3.15] | 20.88 [18.83, 23.03] | +1.05 [+0.70, +1.40] | 34.85 [32.33, 37.42] | +0.42 [+0.11, +0.72] |
| `m1w_xgb_b1` | 0.7091 | 0.2900 | 0.6154 | 3.04 [1.81, 4.60] | 21.84 [19.20, 24.67] | +2.01 [+0.72, +3.54] | 32.52 [29.27, 35.82] | -1.91 [-3.54, -0.11] |
| `m1_xgb_s1` | 0.7046 | 0.3066 | 0.6138 | 2.52 [1.73, 3.52] | 21.31 [19.26, 23.52] | +1.49 [+0.87, +2.18] | 34.07 [31.49, 36.64] | -0.36 [-1.20, +0.54] |
| `m1_xgb_s2` | 0.7044 | 0.3088 | 0.6127 | 3.13 [2.22, 4.23] | 21.16 [19.03, 23.37] | +1.34 [+0.59, +2.11] | 32.96 [30.28, 35.60] | -1.47 [-2.48, -0.43] |
| `m1_xgb_s3` | 0.7016 | 0.3059 | 0.6098 | 2.35 [1.61, 3.24] | 21.58 [19.58, 23.76] | +1.75 [+1.30, +2.23] | 35.74 [33.22, 38.27] | +1.32 [+0.74, +1.91] |
| `m1_xgb_s4` | 0.7047 | 0.3030 | 0.6219 | 3.14 [2.09, 4.37] | 22.01 [19.83, 24.34] | +2.19 [+1.41, +3.03] | 33.81 [31.15, 36.53] | -0.61 [-1.61, +0.41] |
| `m1_xgb_b1_s1` | 0.7158 | 0.3001 | 0.6184 | 2.76 [1.61, 4.32] | 22.63 [20.11, 25.28] | +2.80 [+1.63, +4.12] | 35.12 [32.15, 38.24] | +0.69 [-0.64, +2.07] |
| `m1_xgb_b1_s2` | 0.7160 | 0.3001 | 0.6177 | 3.21 [1.91, 4.95] | 22.62 [20.04, 25.35] | +2.79 [+1.41, +4.30] | 33.88 [30.67, 37.16] | -0.54 [-2.29, +1.28] |
| `m1_xgb_b1_s3` | 0.7174 | 0.2948 | 0.6120 | 3.40 [2.04, 5.17] | 23.04 [20.40, 25.80] | +3.21 [+1.78, +4.73] | 34.42 [31.16, 37.74] | -0.01 [-1.79, +1.76] |
| `m1_xgb_b1_s4` | 0.7202 | 0.3039 | 0.6227 | 3.54 [2.07, 5.34] | 24.04 [21.38, 26.85] | +4.21 [+2.83, +5.74] | 35.17 [32.00, 38.49] | +0.74 [-1.00, +2.50] |
| `m2_lstm` | 0.6832 | 0.2889 | 0.6032 | 3.06 [2.12, 4.25] | 21.32 [19.23, 23.48] | +1.49 [+0.88, +2.16] | 34.03 [31.41, 36.62] | -0.40 [-1.13, +0.44] |
| `m2_lstm_b1` | 0.6950 | 0.2855 | 0.6029 | 3.51 [2.34, 5.05] | 22.07 [19.86, 24.30] | +2.25 [+1.33, +3.14] | 34.09 [31.26, 36.84] | -0.34 [-1.40, +0.73] |
| `m2x_nocum` | 0.6965 | 0.2993 | 0.6120 | 2.39 [1.59, 3.40] | 21.57 [19.56, 23.67] | +1.74 [+1.29, +2.22] | 35.50 [32.88, 38.11] | +1.08 [+0.56, +1.68] |
| `m2x_nocum_b1` | 0.7103 | 0.2921 | 0.6155 | 3.84 [2.47, 5.41] | 23.23 [20.82, 25.72] | +3.40 [+2.18, +4.61] | 34.48 [31.53, 37.40] | +0.05 [-1.46, +1.44] |
| `m3_bert` | 0.7051 | 0.2596 | 0.5883 | 2.13 [1.43, 2.99] | 20.61 [18.72, 22.64] | +0.79 [+0.49, +1.11] | 34.97 [32.57, 37.55] | +0.54 [+0.19, +0.94] |
| `m3_bert_b1` | 0.6927 | 0.2381 | 0.5880 | 2.51 [1.45, 3.79] | 20.08 [17.50, 22.72] | +0.25 [-1.10, +1.66] | 30.76 [27.56, 33.96] | -3.66 [-5.27, -1.93] |

[Discrimination and intervals](tables/discrimination.csv) · [All test operating points](tables/n_summary.csv) · [Matched N=10 cost curves](plots/cost_curves/test_N10.png)

## Baselines and issue-level difficulty

Among the two count rules, `b0_editfail` compared with `b0_err` is 1 higher, 5 lower, 0 tied across 6 matched prefixes with a range of -2.59 to +0.41 pp in overall AUROC. These rules are interpretable and cheap, with discrete scores and tied trajectories. Their usefulness as stop rules depends on which of those ties the threshold acts on and on the resulting success loss.

B1 models share the issue text across runs for that issue. The N-specific B1 score compared with the static score is 4 higher, 2 lower, 1 tied across 7 matched prefixes with a range of -0.86 to +1.86 pp in overall AUROC. Within an issue the score is tied by construction, so issue-level ranking quality can coexist with within-issue AUROC of 0.5. The selected TF–IDF logistic configuration was `{'analyzer': 'char_wb', 'C': 0.1, 'weighting': 'traj', 'min_df': 2, 'max_features': 200000, 'sublinear_tf': True, 'ngram_range': [3, 5]}`.

The same-issue leave-one-out oracle had test AUROC 0.9323–0.9420 over its reported N grid. It uses outcomes of other runs on the same issue and is a retrospective difficulty reference. Its within-issue ordering also reflects the leave-one-out construction. It has no deployable operating point.

[Baseline discrimination](plots/baselines/discrimination.png) · [Baseline policies](plots/baselines/operating_points.png) · [B1 model selection](phases/phase4/b1_grid.csv) · [B1 threshold transfer](phases/phase4/tau_transfer_summary.csv)

## M1 and the supplementary XGBoost grid

Official M1 uses 73 prefix aggregates drawn from command counts and last-command flags, edit and error behavior, repetition, file access, and observation length. The B1 versions add an issue-level score. The logistic model tests additive effects after log scaling and standardization. XGBoost can capture nonlinear interactions and thresholds. Without B1, XGBoost compared with logistic regression is 1 higher, 2 lower, 0 tied across 3 matched prefixes with a range of -0.22 to +4.82 pp in within-issue AUROC at common N. With B1 it is 2 higher, 1 lower, 0 tied across 3 matched prefixes with a range of -0.11 to +5.25 pp.

- `m1_lr_b1` relative to `m1_lr` spans 7 higher, 1 lower, 0 tied across 8 matched prefixes with a range of -0.24 to +2.58 pp in overall AUROC and 5 higher, 3 lower, 0 tied across 8 matched prefixes with a range of -0.49 to +1.07 pp in within-issue AUROC. The combined policy satisfied the loss criterion at N=3, 5, 10, 15, 20, 25, 30.

- `m1_xgb_b1` relative to `m1_xgb` spans 6 higher, 2 lower, 0 tied across 8 matched prefixes with a range of -0.74 to +2.38 pp in overall AUROC and 3 higher, 4 lower, 1 tied across 8 matched prefixes with a range of -1.35 to +0.96 pp in within-issue AUROC. The combined policy satisfied the loss criterion at N=3, 5, 8, 10, 15, 20, 25, 30.

- Top held-out mean absolute SHAP feature for `m1_xgb` by inspected prefix was N=5 `cmd_find_file`, N=10 `n_edit_ok`, N=20 `n_edit_ok`. These ranks describe associations learned by the model. Establishing causal effects would require a separate experiment.

- Top held-out mean absolute SHAP feature for `m1_xgb_b1` by inspected prefix was N=5 `b1_score`, N=10 `b1_score`, N=20 `n_edit_ok`. These ranks describe associations learned by the model. Establishing causal effects would require a separate experiment.

- The registered wider search compared `m1w_xgb` to official `m1_xgb` and was 1 higher, 5 lower, 0 tied across 6 matched prefixes with a range of -0.41 to +0.27 pp in overall AUROC and 4 higher, 2 lower, 0 tied across 6 matched prefixes with a range of -0.41 to +0.29 pp within issue. The comparison uses the same common N cells. The supplementary N=25/30 rows use the much smaller late-running population.

- The registered wider search compared `m1w_xgb_b1` to official `m1_xgb_b1` and was 1 higher, 5 lower, 0 tied across 6 matched prefixes with a range of -1.44 to +0.11 pp in overall AUROC and 3 higher, 3 lower, 0 tied across 6 matched prefixes with a range of -0.59 to +0.54 pp within issue. The comparison uses the same common N cells. The supplementary N=25/30 rows use the much smaller late-running population.

At the extended prefixes, held-out survivor counts and success shares were N=25 with 4,184 trajectories and 4.21% successful, N=30 with 3,284 trajectories and 3.05% successful.

[Official M1 discrimination](plots/m1/discrimination.png) · [Wider grid and extended prefixes](plots/m1_supplementary/discrimination.png) · [Official selected settings](phases/phase5/m1_grid.csv) · [SHAP rank and fold ranges](phases/phase5/shap_importance.csv) · [Retrospective solution-file check](phases/phase5/solution_files_summary.json) · [M1 operating-point transfer diagnostic](phases/phase5/supplementary/tau_transfer_m1_summary.csv)

## S1–S4 and what each change bought

| Stage | Change studied | Frozen choice | Selection metric | Source |
| --- | --- | --- | --- | --- |
| S1 | Recent windows with recency as a candidate | w3 (3-step window) | within-issue AUROC | [OOF candidates](phases/phase5b/selection.csv) |
| S2 | Changes, decay, history weighting and observation-length slopes | sl35 (3- and 5-step observation-length slopes) | within-issue AUROC | [OOF candidates](phases/phase5c/selection.csv) |
| S3 | Thought length and fold-selected lexical counts | length (thought-length features). Vocabulary search selected t=100, K=10 | within-issue AUROC | [OOF candidates](phases/phase5d/selection.csv) |
| S4 | Fit-fold weighting by remaining cost | step_lam1 | combined-policy step savings | [OOF candidates](phases/phase5e/selection.csv) |

S1 tests whether local context adds value over whole-prefix counts. S2 changes the representation of evolution within the prefix. S3 tests whether features derived from the agent's thought text help, with vocabulary estimated separately in each fitting fold. S4 reweights the same S3 feature representation using only fitting-fold costs and labels. The S1–S3 selection criterion aggregates train OOF within-issue AUROC at N=8, 10, 15 and 20. S4 uses train OOF combined-policy step savings at the same prefixes. The final test stage refit the selected variants and checked that their train OOF scores reproduced exactly before scoring each held-out fold.

- `m1_xgb_s1` versus `m1_xgb` increased combined-policy step savings at 4/6 matched prefixes and token savings at 2/6. The within-issue AUROC comparison is 4 higher, 2 lower, 0 tied across 6 matched prefixes with a range of -0.06 to +1.67 pp.

- `m1_xgb_s2` versus `m1_xgb_s1` increased combined-policy step savings at 4/6 matched prefixes and token savings at 3/6. The within-issue AUROC comparison is 3 higher, 2 lower, 1 tied across 6 matched prefixes with a range of -0.12 to +0.48 pp.

- `m1_xgb_s3` versus `m1_xgb_s2` increased combined-policy step savings at 2/6 matched prefixes and token savings at 6/6. The within-issue AUROC comparison is 3 higher, 3 lower, 0 tied across 6 matched prefixes with a range of -0.99 to +0.83 pp.

- `m1_xgb_s4` versus `m1_xgb_s3` increased combined-policy step savings at 5/6 matched prefixes and token savings at 2/6. The within-issue AUROC comparison is 5 higher, 1 lower, 0 tied across 6 matched prefixes with a range of -0.28 to +1.21 pp.

- `m1_xgb_b1_s1` versus `m1_xgb_b1` increased combined-policy step savings at 6/6 matched prefixes and token savings at 4/6. The within-issue AUROC comparison is 5 higher, 1 lower, 0 tied across 6 matched prefixes with a range of -0.00 to +0.46 pp.

- `m1_xgb_b1_s2` versus `m1_xgb_b1_s1` increased combined-policy step savings at 2/6 matched prefixes and token savings at 3/6. The within-issue AUROC comparison is 3 higher, 2 lower, 1 tied across 6 matched prefixes with a range of -0.35 to +0.35 pp.

- `m1_xgb_b1_s3` versus `m1_xgb_b1_s2` increased combined-policy step savings at 2/6 matched prefixes and token savings at 3/6. The within-issue AUROC comparison is 3 higher, 3 lower, 0 tied across 6 matched prefixes with a range of -0.78 to +1.10 pp.

- `m1_xgb_b1_s4` versus `m1_xgb_b1_s3` increased combined-policy step savings at 5/6 matched prefixes and token savings at 3/6. The within-issue AUROC comparison is 4 higher, 2 lower, 0 tied across 6 matched prefixes with a range of -1.56 to +1.16 pp.

The next table uses the available paired intervals. Each cell lists N where the interval is wholly positive / wholly negative. Higher success loss is a worse outcome. Blank results are shown as dashes. Adjacent S1–S3 point comparisons above are descriptive. The paired results cover S4–S3 and each stage against official M1.

| Paired contrast | AUROC +N / −N | Within-issue +N / −N | Success lost +N / −N | Steps saved +N / −N | Tokens saved +N / −N |
| --- | --- | --- | --- | --- | --- |
| `m1_xgb_s1` − `m1_xgb` | 10 / — | — / — | 5 / — | 20 / — | 20 / 5, 10 |
| `m1_xgb_b1_s1` − `m1_xgb_b1` | — / — | — / — | 5 / — | 20 / — | 10, 20 / 5 |
| `m1_xgb_s2` − `m1_xgb` | — / — | — / — | 5, 10 / — | 20 / — | 20 / 5, 10 |
| `m1_xgb_b1_s2` − `m1_xgb_b1` | — / — | — / — | — / — | 10, 20 / — | 10, 20 / 5 |
| `m1_xgb_s3` − `m1_xgb` | 5 / — | — / — | — / — | 10, 20 / — | 5, 10, 20 / — |
| `m1_xgb_b1_s3` − `m1_xgb_b1` | 5 / — | — / — | — / — | 10, 20 / 5 | 10, 20 / 5 |
| `m1_xgb_s4` − `m1_xgb` | — / — | — / — | 5, 10 / — | 5, 10, 20 / — | 20 / 5, 10 |
| `m1_xgb_b1_s4` − `m1_xgb_b1` | — / — | — / — | — / — | 10, 20 / — | 10, 20 / — |
| `m1_xgb_s4` − `m1_xgb_s3` | — / — | 10 / — | 5, 10 / — | 5, 10, 20 / — | 20 / 5, 10 |
| `m1_xgb_b1_s4` − `m1_xgb_b1_s3` | — / — | 10 / — | — / — | 5, 10, 20 / — | 10, 20 / — |

[Official M1 and the full S1–S4 chain](plots/m1_variants/discrimination.png) · [Chain policy outcomes](plots/m1_variants/operating_points.png) · [Paired effect plots](plots/diagnostics/m1_paired_variants.png) · [Exact selected feature and model configurations](phases/m1_variants/selected_configurations.csv)

## M2 and the role of sequence information

The official LSTM combines command embeddings, step-level stats and cumulative prefix stats. The B1 variant supplies the fold-matched issue score to its prediction head. The supplementary no-cumulative models keep step order while removing the provided cumulative summaries. The no-recurrence ablation isolates the contribution of the recurrent architecture in the cumulative-input setting.

- `m2_lstm` selected hidden width 32 and dropout 0.1. Against matched official M1 XGBoost, its within-issue AUROC is 2 higher, 4 lower, 0 tied across 6 matched prefixes with a range of -4.22 to +0.64 pp.

- `m2_lstm_b1` selected hidden width 32 and dropout 0.3. Against matched official M1 XGBoost, its within-issue AUROC is 3 higher, 3 lower, 0 tied across 6 matched prefixes with a range of -5.50 to +0.98 pp.

- Train OOF ablation `m1_lr` averaged 0.6738 overall AUROC and 0.5781 within-issue AUROC over its recorded N grid.

- Train OOF ablation `m1_lr_b1` averaged 0.6981 overall AUROC and 0.5811 within-issue AUROC over its recorded N grid.

- Train OOF ablation `m1_xgb` averaged 0.6881 overall AUROC and 0.5813 within-issue AUROC over its recorded N grid.

- Train OOF ablation `m1_xgb_b1` averaged 0.7093 overall AUROC and 0.5861 within-issue AUROC over its recorded N grid.

- Train OOF ablation `m2_lstm` averaged 0.6853 overall AUROC and 0.5833 within-issue AUROC over its recorded N grid.

- Train OOF ablation `m2_lstm_b1` averaged 0.7057 overall AUROC and 0.5860 within-issue AUROC over its recorded N grid.

- Train OOF ablation `m2x_nocum` averaged 0.6968 overall AUROC and 0.5906 within-issue AUROC over its recorded N grid.

- Train OOF ablation `m2x_norec` averaged 0.6692 overall AUROC and 0.5812 within-issue AUROC over its recorded N grid.

- `m2x_nocum` compared with `m2_lstm` is 6 higher, 0 lower, 0 tied across 6 matched prefixes with a range of +0.35 to +2.66 pp in overall AUROC and 5 higher, 1 lower, 0 tied across 6 matched prefixes with a range of -0.05 to +2.70 pp within issue. Its combined policy passes the loss criterion at N=3, 5, 8, 10, 15, 20.

- `m2x_nocum_b1` compared with `m2_lstm_b1` is 6 higher, 0 lower, 0 tied across 6 matched prefixes with a range of +0.33 to +3.05 pp in overall AUROC and 5 higher, 1 lower, 0 tied across 6 matched prefixes with a range of -0.06 to +3.69 pp within issue. Its combined policy passes the loss criterion at N=8, 15, 20.

| Paired contrast | AUROC +N / −N | Within-issue +N / −N | Success lost +N / −N | Steps saved +N / −N | Tokens saved +N / −N |
| --- | --- | --- | --- | --- | --- |
| `m2_lstm` − `m1_xgb` | — / 10, 15 | — / — | 10 / — | 5 / 3, 15, 20 | 3 / 8, 10, 15 |
| `m2_lstm_b1` − `m1_xgb_b1` | — / 15 | — / 20 | 15, 20 / — | — / 15 | 5, 8 / 3 |

| Paired contrast | AUROC +N / −N | Within-issue +N / −N | Success lost +N / −N | Steps saved +N / −N | Tokens saved +N / −N |
| --- | --- | --- | --- | --- | --- |
| `m2x_nocum` − `m2_lstm` | 10, 15, 20 / — | — / — | — / — | 8, 15, 20 / — | 8, 10, 15, 20 / — |
| `m2x_nocum_b1` − `m2_lstm_b1` | 3, 10, 15, 20 / — | 10 / — | — / 20 | 10, 15, 20 / — | 15, 20 / 3, 5 |
| `m2x_nocum` − `m1_xgb` | — / — | — / — | — / — | 5, 10, 20 / 3 | 3, 15, 20 / — |
| `m2x_nocum_b1` − `m1_xgb_b1` | — / — | — / — | — / — | 10, 15, 20 / — | 8, 10, 15, 20 / 3 |

[Cumulative and no-cumulative comparison](plots/m2/discrimination.png) · [Official M2 paired effects](plots/diagnostics/m2_paired_vs_m1.png) · [No-cumulative paired effects](plots/diagnostics/m2_paired_no_cumulative.png) · [Train OOF ablations](phases/phase6/ablation_oof.csv)

## M3 and how much raw text added

ModernBERT-base consumes the thought, command and clipped observation text from each prefix using a 2,048-token model input budget. It covers only N=5, 10, 20. It trains a shared classifier over those lengths with balanced sampling by issue and keeps the checkpoint with the best within-training validation AUROC. The B1 model combines ModernBERT and B1 logits with a fold-specific late-fusion stacker.

Training loss fell from first to final logged checkpoint in 5/5 folds. Best validation AUROC ranged from 0.6768 to 0.7451. The best validation checkpoint came in epoch one for 4 folds. That pattern indicates a limited validation benefit from later optimization under the registered configuration.

The five ModernBERT training folds consumed 157.0 GPU minutes in total, with 31.0–32.0 minutes per fold before OOF scoring and final evaluation. This training cost is additional to the trajectory context costs measured in the policy tables. [M3 run log](logs/07-m3-dev.log)

- `m3_bert` compared to `m1_xgb` is 2 higher, 1 lower, 0 tied across 3 matched prefixes with a range of -0.41 to +3.41 pp in overall AUROC and 0 higher, 3 lower, 0 tied across 3 matched prefixes with a range of -3.51 to -0.32 pp within issue. Fold-score ensembling changes AUROC by +2.85 to +3.23 pp relative to the average single-fold AUROC.

- `m3_bert_b1` compared to `m1_xgb_b1` is 1 higher, 2 lower, 0 tied across 3 matched prefixes with a range of -2.13 to +0.36 pp in overall AUROC and 0 higher, 3 lower, 0 tied across 3 matched prefixes with a range of -4.65 to -0.30 pp within issue. Fold-score ensembling changes AUROC by +1.63 to +2.18 pp relative to the average single-fold AUROC.

| Paired contrast | AUROC +N / −N | Within-issue +N / −N | Success lost +N / −N | Steps saved +N / −N | Tokens saved +N / −N |
| --- | --- | --- | --- | --- | --- |
| `m3_bert` − `m1_xgb` | 5 / — | — / 10 | — / — | — / 10, 20 | 5 / 20 |
| `m3_bert_b1` − `m1_xgb_b1` | — / — | — / 10 | — / 5 | — / 5, 10, 20 | 5 / 10, 20 |

[M3 discrimination](plots/m3/discrimination.png) · [M3 stopping outcomes](plots/m3/operating_points.png) · [Validation and training curves](plots/diagnostics/m3_validation_and_loss.png) · [M3 paired effects vs M1/M2](plots/diagnostics/m3_paired_models.png) · [Fold and ensemble comparison](phases/phase7/fold_spread.csv)

## Transfer across trajectory generators

- In `x8b_unseen`, `m1_xgb` combined passes the loss criterion at no evaluated prefixes, with a positive K* step interval at no evaluated prefixes and a positive K* token interval at no evaluated prefixes.

- In `x8b_unseen`, `m1_lr` combined passes the loss criterion at no evaluated prefixes, with a positive K* step interval at no evaluated prefixes and a positive K* token interval at no evaluated prefixes.

- In `x405b_unseen`, `m1_xgb` combined passes the loss criterion at no evaluated prefixes, with a positive K* step interval at no evaluated prefixes and a positive K* token interval at no evaluated prefixes.

- In `x405b_unseen`, `m1_lr` combined passes the loss criterion at no evaluated prefixes, with a positive K* step interval at no evaluated prefixes and a positive K* token interval at no evaluated prefixes.

- In `x8b_all`, `m1_xgb` combined passes the loss criterion at no evaluated prefixes, with a positive K* step interval at no evaluated prefixes and a positive K* token interval at no evaluated prefixes.

- In `x8b_all`, `m1_lr` combined passes the loss criterion at no evaluated prefixes, with a positive K* step interval at no evaluated prefixes and a positive K* token interval at no evaluated prefixes.

- In `x405b_all`, `m1_xgb` combined passes the loss criterion at no evaluated prefixes, with a positive K* step interval at no evaluated prefixes and a positive K* token interval at no evaluated prefixes.

- In `x405b_all`, `m1_lr` combined passes the loss criterion at no evaluated prefixes, with a positive K* step interval at no evaluated prefixes and a positive K* token interval at no evaluated prefixes.

- In `test70b`, `m1_xgb` combined passes the loss criterion at N=3, 5, 8, 10, 15, 20, with a positive K* step interval at N=3, 8, 10, 15, 20 and a positive K* token interval at N=10, 15, 20.

- In `test70b`, `m1_lr` combined passes the loss criterion at N=3, 5, 8, 10, 20, with a positive K* step interval at N=3, 5, 8, 10, 20 and a positive K* token interval at N=3, 20.

| Population | Runs | Issues | Success % | Cap | Steps saved % | Tokens saved % | Δ tokens vs K* pp [95%] | Loss upper % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| x8b_unseen | 707 | 87 | 15.70 | K*=40 | 27.81 | 42.53 | — | 22.94 |
| x8b_unseen | 707 | 87 | 15.70 | Proposal K=35 | 32.40 | 49.37 | +6.84 [+6.02, +7.89] | 29.23 |
| x405b_unseen | 255 | 22 | 15.69 | K*=40 | 29.03 | 42.94 | — | 10.00 |
| x405b_unseen | 255 | 22 | 15.69 | Proposal K=35 | 33.96 | 50.31 | +7.37 [+6.19, +8.74] | 10.00 |
| x8b_all | 4,053 | 528 | 15.15 | K*=40 | 27.44 | 42.24 | — | 15.80 |
| x8b_all | 4,053 | 528 | 15.15 | Proposal K=35 | 32.14 | 49.31 | +7.07 [+6.69, +7.50] | 18.88 |
| x405b_all | 1,191 | 100 | 25.86 | K*=40 | 29.12 | 44.25 | — | 10.48 |
| x405b_all | 1,191 | 100 | 25.86 | Proposal K=35 | 33.78 | 51.15 | +6.90 [+6.19, +7.62] | 13.31 |
| test70b | 12,481 | 602 | 16.70 | K*=40 | 19.83 | 34.43 | — | 2.90 |
| test70b | 12,481 | 602 | 16.70 | Proposal K=35 | 23.46 | 40.68 | +6.26 [+5.95, +6.57] | 4.13 |

The issue-unseen subsets isolate shifts to different Llama sizes on issues outside the model's 70B training issues. The all-issues subsets measure a broader transfer setting with possible training-issue overlap. The smaller subsets have fewer independent issues and less precise loss estimates. The 70B operating points are reused unchanged, so transfer is evidence about portability of the policy as trained.

[Transfer discrimination and costs](plots/cross_model/transfer_combined.png) · [Transfer paired cost and loss plots](plots/diagnostics/cross_model_paired_vs_k.png) · [Transfer by model, policy and N](phases/phase8/n_summary.csv)

## Coverage, uncertainties and reporting

The table shows the entire prespecified prefix sweep. `×` means the 5% loss upper-bound criterion failed. `✓` means it passed with no positive paired savings interval against K*. `S` and `T` denote gains in steps and tokens respectively and `B` denotes gains in both. These are measured cells under the frozen thresholds, prefixes and model specifications.

| Model | Inputs or variant | Single policy by N | Combined policy by N |
| --- | --- | --- | --- |
| `b0_err` | Prefix error-count heuristic | 3:✓ 5:✓ 8:✓ 10:✓ 15:✓ 20:✓ | 3:✓ 5:B 8:B 10:B 15:B 20:B |
| `b0_editfail` | Prefix rejected-edit count | 3:✓ 5:✓ 8:✓ 10:✓ 15:✓ 20:✓ | 3:✓ 5:✓ 8:B 10:B 15:B 20:B |
| `b1_static` | Issue-only TF–IDF logistic model fit at N=0 | 3:× 5:× 8:× 10:× 15:× 20:✓ | 3:× 5:✓ 8:× 10:× 15:× 20:S |
| `b1_n` | Issue-only TF–IDF logistic model fit per N | 3:× 5:× 8:× 10:× 15:✓ 20:✓ | 3:× 5:✓ 8:✓ 10:× 15:✓ 20:✓ |
| `m1_lr` | Logistic regression on prefix aggregates | 3:× 5:× 8:× 10:× 15:× 20:B 25:B 30:B | 3:B 5:S 8:S 10:S 15:× 20:B 25:B 30:B |
| `m1_lr_b1` | Logistic regression on aggregates and B1 | 3:× 5:× 8:× 10:× 15:× 20:B 25:B 30:B | 3:B 5:S 8:× 10:S 15:B 20:B 25:B 30:B |
| `m1_xgb` | Gradient-boosted trees on prefix aggregates | 3:× 5:× 8:× 10:✓ 15:B 20:B 25:B 30:B | 3:S 5:✓ 8:S 10:B 15:B 20:B 25:B 30:B |
| `m1_xgb_b1` | Gradient-boosted trees on aggregates and B1 | 3:× 5:× 8:× 10:× 15:B 20:B 25:B 30:B | 3:B 5:✓ 8:✓ 10:S 15:B 20:B 25:B 30:B |
| `m1w_xgb` | XGBoost with the supplementary shallow grid | 3:× 5:× 8:× 10:× 15:S 20:B | 3:B 5:✓ 8:S 10:B 15:B 20:B |
| `m1w_xgb_b1` | Supplementary XGBoost grid with B1 | 3:× 5:× 8:× 10:× 15:× 20:B | 3:S 5:× 8:✓ 10:S 15:B 20:B |
| `m1_xgb_s1` | w3 | 3:× 5:× 8:× 10:× 15:B 20:B | 3:S 5:✓ 8:S 10:S 15:B 20:B |
| `m1_xgb_s2` | sl35 | 3:× 5:× 8:× 10:× 15:B 20:B | 3:S 5:✓ 8:S 10:S 15:B 20:B |
| `m1_xgb_s3` | length | 3:× 5:× 8:× 10:× 15:B 20:B | 3:S 5:B 8:S 10:B 15:B 20:B |
| `m1_xgb_s4` | step_lam1 | 3:× 5:× 8:× 10:× 15:B 20:B | 3:S 5:S 8:S 10:S 15:B 20:B |
| `m1_xgb_b1_s1` | w3 | 3:× 5:× 8:× 10:× 15:B 20:B | 3:B 5:× 8:S 10:S 15:B 20:B |
| `m1_xgb_b1_s2` | sl35 | 3:× 5:× 8:× 10:× 15:B 20:B | 3:B 5:× 8:S 10:S 15:B 20:B |
| `m1_xgb_b1_s3` | length | 3:× 5:× 8:× 10:× 15:B 20:B | 3:B 5:✓ 8:S 10:× 15:B 20:B |
| `m1_xgb_b1_s4` | step_lam1 | 3:× 5:× 8:× 10:× 15:B 20:B | 3:B 5:× 8:× 10:× 15:B 20:B |
| `m2_lstm` | LSTM with cumulative prefix summaries | 3:× 5:× 8:× 10:× 15:× 20:B | 3:B 5:S 8:S 10:S 15:× 20:B |
| `m2_lstm_b1` | Cumulative LSTM with B1 at the prediction head | 3:× 5:× 8:× 10:× 15:× 20:B | 3:✓ 5:S 8:S 10:× 15:× 20:B |
| `m2x_nocum` | LSTM with step-level inputs and no cumulative summaries | 3:× 5:× 8:× 10:× 15:× 20:B | 3:B 5:S 8:S 10:B 15:B 20:B |
| `m2x_nocum_b1` | LSTM without cumulative summaries, with B1 at the head | 3:× 5:× 8:× 10:× 15:× 20:B | 3:× 5:× 8:S 10:× 15:B 20:B |
| `m3_bert` | ModernBERT on prefix text | 5:× 10:✓ 20:S | 5:B 10:B 20:B |
| `m3_bert_b1` | ModernBERT and B1 late-fusion logistic stacker | 5:✓ 10:✓ 20:S | 5:✓ 10:✓ 20:B |
| `ref_oracle_loo` | Leave-one-out same-issue reference | Reference | Reference |

For the single policy, 58 measured model–prefix cells passed the loss criterion. Paired gains versus K* were supported for both costs in 36 cells, for steps alone in 3, and for tokens alone in 0. Models and prefixes in that sweep share trajectories.

For the combined policy, 125 measured model–prefix cells passed the loss criterion. Paired gains versus K* were supported for both costs in 69 cells, for steps alone in 37, and for tokens alone in 0. Models and prefixes in that sweep share trajectories.

Interpreting cost claims requires the loss interval and both savings columns. A gain in steps can coincide with flat or negative token savings because stopping different trajectories removes different amounts of accumulated context. The score-based policy is a triage rule for the setting and model family measured here. The source dataset, issue overlap in some transfer populations, prefix-dependent coverage, and the use of a cost proxy all limit how directly it translates to another deployment.

The run passed 255 repository tests and completed the phase criteria. Phase 5d's synthetic test checkpoints were moved out of the production cache before its bootstrap so the full run used real issue resamples. All 48 Phase 5d bootstrap task outputs carried the 1,000-draw count. No second final run was made for this report.

[Run provenance](manifest.json) · [Checkpoint isolation audit](logs/checkpoint-audit.log) · [Plot guide](PLOT_INDEX.md) · [Machine-readable operating points](tables/operating_points.json) · [Machine-readable model scores](tables/discrimination.json)
