# Plot guide

Compare models at the same prefix length and on the same evaluation population. The train and held-out panels use separate estimates. CI bars and bands are 95% issue-bootstrap intervals. Operating-point and cost plots distinguish step savings from context-token savings. For the two policy modes use the complete tables and the held-out outcome matrix.

Read the [final analysis](ANALYSIS.md) alongside the [discrimination](tables/discrimination.csv) and [operating-point](tables/operating_points.csv) tables.

## All models

- [Discrimination test](plots/all_models/discrimination_test.png)
- [Discrimination train oof](plots/all_models/discrimination_train_oof.png)
- [Operating points combined](plots/all_models/operating_points_combined.png)
- [Operating points single](plots/all_models/operating_points_single.png)

## Baseline comparisons

- [Discrimination](plots/baselines/discrimination.png)
- [Operating points](plots/baselines/operating_points.png)

## Official M1

- [Discrimination](plots/m1/discrimination.png)
- [Operating points](plots/m1/operating_points.png)

## Official M1 and the selected S1–S4 chain

- [Discrimination](plots/m1_variants/discrimination.png)
- [Operating points](plots/m1_variants/operating_points.png)

## Wider M1 grid and extended prefixes

- [Discrimination](plots/m1_supplementary/discrimination.png)
- [Operating points](plots/m1_supplementary/operating_points.png)

## LSTM with and without cumulative inputs

- [Discrimination](plots/m2/discrimination.png)
- [Operating points](plots/m2/operating_points.png)

## ModernBERT base and late fusion

- [Discrimination](plots/m3/discrimination.png)
- [Operating points](plots/m3/operating_points.png)

## Cross-model transfer

- [Transfer combined](plots/cross_model/transfer_combined.png)

## Pairing, ablations, coverage and training

- [All models outcome matrix](plots/diagnostics/all_models_outcome_matrix.png)
- [Cross model paired vs k](plots/diagnostics/cross_model_paired_vs_k.png)
- [M1 paired variants](plots/diagnostics/m1_paired_variants.png)
- [M2 paired no cumulative](plots/diagnostics/m2_paired_no_cumulative.png)
- [M2 paired vs m1](plots/diagnostics/m2_paired_vs_m1.png)
- [M3 paired models](plots/diagnostics/m3_paired_models.png)
- [M3 validation and loss](plots/diagnostics/m3_validation_and_loss.png)
- [Phase5b oof ablation](plots/diagnostics/phase5b_oof_ablation.png)
- [Phase5c oof ablation](plots/diagnostics/phase5c_oof_ablation.png)
- [Phase5d oof ablation](plots/diagnostics/phase5d_oof_ablation.png)
- [Phase5e oof ablation](plots/diagnostics/phase5e_oof_ablation.png)
- [Population by prefix](plots/diagnostics/population_by_prefix.png)

## Matched prefix cost curves

- [Test n10](plots/cost_curves/test_N10.png)
- [Test n15](plots/cost_curves/test_N15.png)
- [Test n20](plots/cost_curves/test_N20.png)
- [Test n3](plots/cost_curves/test_N3.png)
- [Test n5](plots/cost_curves/test_N5.png)
- [Test n8](plots/cost_curves/test_N8.png)

## Original phase figures

- [M1 SHAP summaries, dependence plots, and the project AUROC figure](phases/phase5/figures/)

- Selection data and exported threshold-cost curves are described in [the run guide](README.md).
