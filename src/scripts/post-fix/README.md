# Post-fix reproduction of PAS Tables 2, 3, and Figure 3

Re-runs the PAS paper's benchmarks after the code corrections found by comparing
the implementation against the camera-ready appendices. Produces the paper's
Table 3 (real data), Table 2 (synthetic model), and Figure 3 (synthetic-model
power-tuning and adaptive-shrinkage parameters). It also produces a diagnostic
plot of the per-problem PT variances that determine Figure 3's localized
shrinkage weights, plus a broken-axis rendering of the shrinkage weights so
their within-predictor trends remain visible. The tables include their published
counterparts for reference.

## Reproduce

```bash
./src/scripts/post-fix/reproduce.sh          # K = 200 (a few minutes, 5 subprocesses)
TRIALS=5 ./src/scripts/post-fix/reproduce.sh # smoke run
```

Or the two steps separately:

```bash
uv run python src/scripts/post-fix/run_post_fix_tables.py --trials 200 --num-workers 5
uv run python src/scripts/post-fix/make_latex_tables.py
```

Figure 3 can also be regenerated independently (it does not require the
real-world data files):

```bash
uv run python src/scripts/paper_plots/plot_synthetic_params.py
```

Real-data configuration matches Appendix E.4: `K = 200` replicates, dataset
default split seed (42; replicate *k* uses seed 42 + *k*),
`train_test_split = 0.2`. Synthetic configuration matches Table 2 / Figure 3:
`m = 200`, `n_j = 20`, `N_j = 80`, covariate standard deviation `psi = 0.2`,
additional response variance `c = 0.05`, and split seed 4321. The five dataset cells run as independent subprocesses; each
builds its own dataset rather than pickling one across the process boundary.

Second moments are per problem (Appendix C.1) on the real datasets, reported in
the tables as `share_var=False` — the configuration behind the paper's numbers.
On the synthetic model the true moments are evaluated from the corrected closed
forms (Appendix E.1), rather than approximated with Monte Carlo draws, so the
choice does not arise there.

The PT, shrink-average, and PAS rows use the unrestricted per-problem
`lambda_j^*` in Eq. (13). The general estimator API retains clipping as a
defensive default, so the reproduction scripts pass `clip_lambda=False`
explicitly. UniPT and UniPAS are intentionally different: their one shared
power-tuning parameter remains clipped to `[0, 1]`, as specified in Appendix
C.2.

To rerun only the two synthetic cells while preserving the existing real-data
results in the combined output, use:

```bash
uv run python src/scripts/post-fix/run_post_fix_tables.py \
  --datasets synthetic_f1 synthetic_f2 --preserve-unselected --num-workers 2
uv run python src/scripts/post-fix/make_latex_tables.py
```

## Outputs

| File | Contents |
|------|----------|
| `post_fix_tables.pdf` | Standalone 2-page render: post-fix Tables 3 and 2 + note, published counterparts on p. 2 |
| `post_fix_tables.tex` | Source for the above |
| `post_fix_tables_bodies.tex` | Bare `table` environments, for pasting into the paper |
| `../../../images/synthetic_params.pdf` | Post-fix Figure 3, with 200 problems for each predictor |
| `../../../images/synthetic_omegas_zoomed.pdf` | Broken-axis close-up of Figure 3's problem-level adaptive shrinkage weights |
| `../../../images/synthetic_variances.pdf` | Diagnostic plot of the per-problem $\tilde\sigma_j^2=\operatorname{Var}(\hat\theta_j^{\mathrm{PT}})$ values used to localize shrinkage |
| `results/synthetic_params.csv` | The 400 plotted rows and their $\eta_j$, $\hat\lambda_j$, $\hat\omega_j$, and $\tilde\sigma_j^2$ values |
| `results/synthetic_variances.csv` | The four columns used by the PT-variance diagnostic plot |
| `results/summary.csv` | One row per (dataset, estimator) with mean and SE |
| `results/raw_*.csv` | Per-replicate MSE and %-improved values |
| `results/meta.json` | Run configuration |

## The six fixes

All verified against the camera-ready appendices; regression tests in
`tests/test_estimator_fixes.py`.

1. **`get_shrinkage_to_mean_estimators` target.** Shrank toward the per-problem
   prediction mean `f_x_bar` while its SURE objective was built around the grand
   mean of the PT estimates. Now shrinks toward `grand_mean`, per Eq. (29) and
   Algorithm 4 line 14.
2. **UniPAS variance proxy.** Collapsed the per-problem variance to one scalar
   `mean(compound_pt_vars)`. The paper averages the *moments* and then divides by
   problem-specific `n_j, N_j`, so σ̌_j² stays a vector (Eq. 25). The CURE
   objective separately needs the fully per-problem σ̇_j².
3. **UniPAS covariance term.** Used `Cov(Y,f)/(N_j n_j)`; Eq. (26) requires
   `γ̇_j = λ̂_clip · τ̂_j²/N_j` — a different moment, an extra `λ̂` factor and a
   different denominator.
4. **The moment choice ignored by the second moments.** In `get_pas_estimators`
   and `get_shrinkage_to_mean_estimators` the flag reached only the PT lambdas;
   the moments were always pooled. `get_shrinkage_only_estimators` had the flag
   *inverted* relative to its docstring and to `get_pt_ppi_estimators`. All three
   now share one `estimate_second_moments()` helper following Appendix C.1.
5. **UniPT λ not clipped.** Appendix C.2 defines
   `λ̂_clip := clip(λ̂, [0, 1])`; the code used the raw ratio.
6. **Pseudo ground-truth.** Amazon and Galaxy averaged only the *unlabelled*
   split. Appendix E.4 defines `θ̇_j := T_j⁻¹ Σ_i Ẏ_ij`, the mean of **all**
   responses (as LM Arena already did). This also makes the estimand fixed rather
   than re-drawn each replicate.

Two changes beyond that list, both consequences of the above:

- **`MIN_PT_VARIANCE` floor.** Fix 4 exposed that the plug-in PT variance
  `σ²/n + λ²τ²(n+N)/(nN) − 2λγ/n` is a difference of sample moments and goes
  negative on 3 of 200 Amazon-tuned problems once moments are per-problem, which
  would push `ω_j` above 1 and shrink *away* from the target. Flooring sends
  `ω_j → 1`, leaving those problems unshrunk.
- **`τ̂²` over all `n+N` predictions** in `get_pas_estimators`, which previously
  used the unlabelled ones only. Appendix C.1 defines it over `n_j + N_j` points
  and PT/UniPAS already did this.

Not changed: Appendix C.1 centres `f` at `Z̄_j^{N+n}` when forming γ̂_j, while the
code uses `np.cov` (centring at the labelled mean). Since `Σ_i(Y_ij − Ȳ_j) = 0`,
γ̂_j does not depend on that centring constant at all, so the two expressions are
algebraically identical.

## Reading the results

**The real-data tables are not directly comparable row-for-row with the published
Table 3.** Fix 6 changes the estimand from the mean of the unlabelled responses
to the mean of all responses, so every estimator is evaluated against a
different pseudo ground-truth.

**The synthetic table also has a different configuration.** Its target remains
known exactly as `θ_j = η_j²`, but the revised experiment changes `psi` from
0.1 to 0.2. The published Table 2 is therefore retained only as a reference.

**Which estimators take a moment argument.** PT, Shrink Classical, Shrink Avg and
PAS do: the paper assumes their second moments known, so the plug-in used in
practice is an implementation choice it does not specify. PT, Shrink Avg, and
PAS also take `clip_lambda`; the reproduction uses the unrestricted Eq. (13)
setting. UniPT and UniPAS deliberately expose neither argument — Appendix
C.2/C.3 and Algorithm 2 fix their moment handling and clip their shared
power-tuning parameter, making those choices part of the estimator's definition
rather than knobs. Classical, Prediction Avg and PPI have no second-moment
estimate at all.
