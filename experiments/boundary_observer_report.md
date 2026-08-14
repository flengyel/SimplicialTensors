# Raw boundary observer study

**Verdict:** This controlled run is a negative screen for DSTM-specific neural introspection; the raw boundary was below the mean matched-random observer on every primary forecast and intervention endpoint.

## Question and design

This experiment tests whether the unmodified DSTM boundary of a tied square residual/recurrent weight is a useful observer of its training dynamics. The boundary is not added to the objective or optimizer. Shadow interventions are evaluated on copied weights and are never committed to a trajectory.

The digits classifier reuses one square hidden-state map at every residual step. Whole initialization trajectories, rather than individual epochs, are assigned to ridge fitting, hyperparameter tuning, or final evaluation. Each random control has the same rank and exactly the same nonzero singular values as the DSTM boundary. It also maps into the DSTM boundary range, so applying the next boundary gives zero. The controls therefore preserve the finite-chain property while randomizing which weight-space directions are observed.

Boundary outputs are expressed in orthonormal coordinates of their range; this discards only identically zero/redundant output directions. Baseline features are epoch, current training loss, and weight norm. The primary forecast endpoints are future training- and validation-loss changes. The DSTM boundary-gradient and syndrome-velocity forecasts are secondary because their targets are defined by DSTM itself. A positive screen requires each future-loss forecast to beat the scalar baseline and reach at least the 90th matched-control percentile. Each shadow endpoint must reach that percentile and have a positive lower bound for its paired 95% interval against the random-control mean.

## Primary results

Held-out standardized-coordinate future-loss forecast \(R^2\):

| Target | Baseline | DSTM | Random mean | Random best | DSTM − random mean | Rank | Percentile |
|---|---:|---:|---:|---:|---:|---:|---:|
| Future training-loss change | 0.9726 | 0.8408 | 0.9137 | 0.9690 | -0.0729 | 15/17 | 12.5% |
| Future validation-loss change | 0.9736 | 0.8491 | 0.8669 | 0.9671 | -0.0178 | 14/17 | 18.8% |

Norm-matched held-out shadow improvements per unit weight perturbation, summarized after first averaging within each held-out trajectory:

| Evaluated loss | DSTM visible | DSTM null | Visible − null | DSTM visible − random mean (95% paired CI) | Rank | Percentile |
|---|---:|---:|---:|---:|---:|---:|
| Training | 0.0468231 | 0.0671702 | -0.020347 | -0.00868572 (-0.0115772, -0.00579426) | 17/17 | 0.0% |
| Validation | 0.0375144 | 0.0554731 | -0.0179588 | -0.00758286 (-0.012613, -0.00255271) | 17/17 | 0.0% |

Across trajectories, final validation accuracy averaged 0.9243; final test accuracy averaged 0.9148. These task metrics establish that the recorded paths are learning trajectories; they are not a comparison of training methods because every path uses the same method.

## Secondary DSTM-defined forecasts

Held-out standardized-coordinate forecast \(R^2\):

| Target | Baseline | DSTM | Random mean | DSTM − random mean |
|---|---:|---:|---:|---:|
| Current DSTM boundary gradient | -0.0618 | -0.0601 | -0.0662 | +0.0060 |
| Future DSTM syndrome velocity | -0.1102 | -0.0990 | -0.1146 | +0.0156 |

## Validity checks

- DSTM boundary rank: 45 of 100 weight-space dimensions.
- Largest matched-control singular-value error: 1.33e-15.
- Largest matched-control next-boundary operator error: 3.47e-16.
- Manual recurrent-weight gradient finite-difference relative error: 3.12e-09.
- Exact DSTM visible/null projector agreement: at most 1.16e-16 entrywise.

## Interpretation

No privileged task information was detected in this setting. DSTM ranked 15/17 and 14/17 on future training- and validation-loss forecasts and 17/17 on both shadow endpoints. It was below the scalar baseline on both forecasts and below the mean matched-random observer on every primary endpoint. This run does not justify adding a learned feedback controller.

The boundary-gradient and syndrome-velocity targets are DSTM-defined targets, so success on them alone cannot establish privileged task information. The future-loss target and matched shadow comparison are the relevant guards against that circularity. The visible and null DSTM subspaces have dimensions 45 and 55, so their direct contrast is not rank-matched; the DSTM-visible versus random-visible comparison is. Results concern a linear ridge observer, one coordinate convention, one small dataset, and one tied architecture. Hidden coordinates retain their index labels but are not semantically aligned across independent initializations; the held-out-trajectory test therefore measures portability rather than ruling out a controller fitted within one trajectory. The random-control sample is finite. Forecast R² values are pooled over the five held-out trajectories and have no trajectory-level confidence intervals. Only the shadow comparison reports paired trajectory-level intervals, so forecast uncertainty is not fully quantified.

Both secondary DSTM-defined forecasts had negative held-out R² despite ranking above the matched controls. Their relative rank therefore does not show useful prediction.

## Reproduction

From the repository root, with project dependencies installed:

```bash
PYTHONPATH=src python experiments/boundary_observer_study.py
```

Machine-readable outputs are in `experiments/results/boundary_observer_study/`. `summary.json` records the full configuration, trajectory split, operator and gradient checks, aggregates, runtime, and software versions.
