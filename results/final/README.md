# Full experiment results

- `tables/` contains unified 70B discrimination, operating points, and all-prefix summaries in CSV and JSON.
- `plots/` contains all-model, model-family, cost-curve, and cross-model transfer figures.
- `phases/` contains original per-phase reports and their train-only candidate tables, SHAP figures, and paired comparisons.
- `phases/m1_variants/` contains the final evaluation for the selected S1–S4 chain and the comparisons paired against official M1.
- `logs/` contains stage output. `manifest.json` records the run fingerprint, protocol, devices, and working paths.

The 70B training rows use out-of-fold predictions. Test discrimination uses fold-mean scores. Test operating points apply the train-selected policy separately to each fold-model score and average the results. The transfer experiment uses the separate populations named in its tables.

Steps and context-token savings are fractions in the tables. Differences are relative to the train-selected fixed cap K*. The proposal cap is also shown. Both fixed caps appear in the cost curves. Shown intervals are 95% issue-bootstrap intervals where available. A blank CSV field or null JSON value represents an unavailable or non-finite entry. An infinite threshold in CSV means the score does not trigger early stopping.

S1–S4 test figures use the bundles selected by this run's train OOF stages. Unselected S1–S4 candidates and the other development ablations appear in their phase's OOF tables.
