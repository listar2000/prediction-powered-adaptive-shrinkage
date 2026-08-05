# Post-fix reproduction of PAS Table 3

Re-runs the PAS paper's real-data benchmark after six code corrections found by
comparing the implementation against the camera-ready appendices. Produces the
paper's Table 3 twice — once with second moments estimated per problem
(`share_var=False`) and once pooled across problems (`share_var=True`).

## Reproduce

```bash
./src/scripts/post-fix/reproduce.sh          # K = 200 (a few minutes, 6 subprocesses)
TRIALS=5 ./src/scripts/post-fix/reproduce.sh # smoke run
```

Or the two steps separately:

```bash
uv run python src/scripts/post-fix/run_post_fix_tables.py --trials 200 --num-workers 6
uv run python src/scripts/post-fix/make_latex_tables.py
```

Configuration matches Appendix E.4: `K = 200` replicates, dataset default split
seed (42; replicate *k* uses seed 42 + *k*), `train_test_split = 0.2`. The six
`(dataset, share_var)` cells run as independent subprocesses; each builds its own
dataset rather than pickling one across the process boundary.

## Outputs

| File | Contents |
|------|----------|
| `post_fix_tables.pdf` | Standalone 2-page render: Tables 1–2 (post-fix) + note, published Table 3 for reference on p. 2 |
| `post_fix_tables.tex` | Source for the above |
| `post_fix_tables_bodies.tex` | Bare `table` environments, for pasting into the paper |
| `results/summary.csv` | One row per (dataset, share_var, estimator) with mean and SE |
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
4. **`share_var` ignored by the second moments.** In `get_pas_estimators` and
   `get_shrinkage_to_mean_estimators` the flag reached only the PT lambdas; the
   moments were always pooled. `get_shrinkage_only_estimators` had the flag
   *inverted* relative to its docstring and to `get_pt_ppi_estimators`. All three
   now share one `estimate_second_moments()` helper following Appendix C.1, with
   `True` = pooled and `False` = per-problem throughout.
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
code uses `np.cov` (centring at the labelled mean). Both are consistent
estimators of the same quantity.

## Reading the results

**The MSE scale changed — these tables are not comparable row-for-row with the
published Table 3.** Fix 6 changes the estimand. Since
`Ȳ_lab − Ȳ_all = (N/T)(Ȳ_lab − Ȳ_unlab)` pointwise, every squared error is
rescaled by `(N/T)² = 0.641` at the 80/20 split; the measured Classical ratio is
0.6417. Relative orderings carry over, absolute levels do not.

**Only four rows respond to `share_var`**: PT, Shrink Classical, Shrink Avg, PAS.
Classical / Prediction Avg / PPI have no second-moment estimate to share, and
UniPT / UniPAS deliberately omit the flag (the paper notes sharing variance is not
meaningful once a single global λ is used). The runner asserts those five rows
match to 1e-12 across the two runs, which doubles as a check on the parallel
harness.

**Two findings worth attention.** UniPAS now has the lowest MSE on all three
datasets (5.475 / 2.071 / 0.546 against PAS's 5.588 / 2.758 / 0.581), reversing
the published ordering — so the paper's claim that "PAS achieves the lowest MSE
among all estimators" no longer holds. And Shrink Avg is now genuinely distinct
from Shrink Classical, since before fix 1 both shrank toward the same target.
