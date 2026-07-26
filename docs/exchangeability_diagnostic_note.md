# Target-dependence diagnostic for bias exchangeability

A permutation-calibrated Wasserstein test and its LM Arena visualization.

---

## 1. Question addressed

The rebiasing model assumes that task biases are draws from a common law,

$$
b_i \overset{\mathrm{iid}}{\sim} G,
\qquad
G \text{ does not depend on the fixed target } \theta_i.
$$

With only one latent bias per task, unrestricted exchangeability is not
directly testable. The diagnostic instead tests the concrete implication that
the *distribution of bias is invariant across target levels*: do tasks with
low, medium, and high target values exhibit visibly different bias
distributions, and how large is the discrepancy?

### Pseudo-oracle task summaries

For LM Arena task $i$ (one model pair) with full-corpus human outcomes
$Y_{ij} \in \{0,1\}$ and Bradley–Terry reward-model probabilities
$\hat{p}_{ij}$, define

$$
\tilde{\theta}_i = \frac{1}{n_i}\sum_{j=1}^{n_i} Y_{ij},
\qquad
\tilde{b}_i = \frac{1}{n_i}\sum_{j=1}^{n_i} \hat{p}_{ij} - \tilde{\theta}_i.
$$

Following the paper's real-data evaluation convention, the full-corpus mean is
treated as pseudo-ground truth. The test conditions on
$(\tilde{\theta}_i, \tilde{b}_i)$ and does not attempt to separate
pseudo-oracle error from latent-bias variation.

---

## 2. Statistic and permutation calibration

Sort the $M$ tasks by $\tilde{\theta}_i$ and form $K=4$ approximately
equal-count bins $Q_1,\dots,Q_4$. Let

$$
\hat{F}_q = \frac{1}{|Q_q|}\sum_{i \in Q_q} \delta_{\tilde{b}_i}
$$

be the empirical pseudo-bias distribution in bin $q$. For each pair $q < r$,
compute the one-dimensional 1-Wasserstein distance

$$
D_{qr} = W_1(\hat{F}_q, \hat{F}_r)
= \int_{-\infty}^{\infty} \left|\hat{F}_q(t) - \hat{F}_r(t)\right| \mathrm{d}t.
$$

The omnibus statistic is

$$
T_{\mathrm{obs}} = \frac{\max_{q<r} D_{qr}}{s_{\tilde{b}}},
\qquad
s_{\tilde{b}}^2 = \frac{1}{M-1}\sum_i \left(\tilde{b}_i - \bar{\tilde{b}}\right)^2 .
$$

This gives an interpretable effect size: $T = 0.4$ means the most separated
target bins differ by 0.4 overall bias standard deviations in Wasserstein
distance. Because $W_1$ compares the entire empirical distributions, it can
detect changes in location, spread, skewness, or multimodality.

Under the null that pseudo-bias labels are exchangeable with respect to the
fixed target bins, randomly permute the $\tilde{b}_i$ across tasks. The
*maximum* over all six bin pairs is recomputed for every permutation, so
selection of the most different pair is included in the calibration. With $B$
permutations,

$$
\hat{p} = \frac{1 + \sum_{b=1}^{B} \mathbf{1}\{T^{(b)} \ge T_{\mathrm{obs}}\}}{B+1}.
$$

This is a custom omnibus statistic combined with the classical Fisher–Pitman
randomization principle; it is not presented as a canonical named
"exchangeability test." The $+1$ correction prevents zero Monte Carlo
p-values.

- **Null:** the pseudo-bias distribution is invariant across the four fixed
  target bins.
- **Alternative:** at least one pair of target bins has a different
  pseudo-bias distribution.
- **Reported together:** a permutation p-value for evidence and $T$ for
  magnitude.

---

## 3. LM Arena result using `clean_summary_v2.csv`

The analysis uses $M = 298$ model-pair tasks, $B = 9{,}999$ permutations, and
random seed 8273. The v2 artifact contains 3,544 distinct Bradley–Terry
probabilities over $[0,1]$ (rather than the binary prediction artifact).

| Quantity | Value |
| --- | --- |
| Target-bin sizes | 75, 74, 75, 74 |
| Overall pseudo-bias mean / SD | −0.03667 / 0.11333 |
| Largest pairwise difference | $Q_2$ versus $Q_3$ |
| Raw maximum $W_1$ distance | 0.04656 |
| Standardized statistic $T_{\mathrm{obs}}$ | 0.41082 |
| Permutation-null mean / SD | 0.28126 / 0.08109 |
| Observed statistic in null-SD units | 1.598 |
| Monte Carlo p-value | 0.0709 |

The result indicates a **moderate, borderline target-dependent departure**: the
largest distributional separation is about 0.41 bias SD, and the permutation
p-value lies between 0.05 and 0.10. The third target quartile has a less
negative mean bias (−0.0079) than the second (−0.0541), which drives the
largest $W_1$ separation. This does *not* establish exact exchangeability;
rather, it quantifies the observed target-related violation and shows that it
is much smaller than one full bias standard deviation.

---

## 4. Software design and commands

The integration introduces a small statistical-testing layer parallel to the
estimator and CI registries:

- `pas.statistical_tests.base`: shared `StatisticalTest` protocol and
  `StatisticalTestResult` with standardized formatting and JSON output.
- `pas.statistical_tests`: registry plus `run_statistical_test(...)`.
- `pas.statistical_tests.exchangeability`: LM Arena aggregation, balanced
  bins, Wasserstein statistic, and permutation calibration.
- `pas.statistical_tests.plotting`: generic permutation-null plot, the
  descriptive bias-versus-target panel, and the three-panel figure.
- `src/scripts/run_lmarena_exchangeability.py`: reproducible command-line
  entry point.

Full diagnostic (test plus the three-panel figure; the permutation loop takes
roughly a minute):

```bash
uv run python src/scripts/run_lmarena_exchangeability.py \
  --csv data/lmarena/clean_data/clean_summary_v2.csv \
  --bins 4 --permutations 9999 --seed 8273 \
  --out-dir results/lmarena_exchangeability
```

This writes `diagnostics.json`, `task_summary.csv`, and PNG/PDF versions of
`exchangeability_diagnostic`.

Panel A only (bias versus target with quartile means ± 1.96 SE). This panel is
purely descriptive, so `--panel a` skips the permutation test entirely and
finishes in a few seconds:

```bash
uv run python src/scripts/run_lmarena_exchangeability.py --panel a
```

It writes `exchangeability_panel_a.png` and `exchangeability_panel_a.pdf` to
the output directory.

---

## 5. Scope and caveats

- The diagnostic targets $b_i \perp \theta_i$ through coarse target bins; it
  does not detect every possible nonexchangeable structure (for example,
  model-identity effects within LM pairs).
- The pseudo-oracle treatment ignores within-task uncertainty and the
  mechanical reuse of $Y_{ij}$ in $\tilde{\theta}_i$ and $\tilde{b}_i$. A
  noise-aware extension could bootstrap $\tilde{b}_i \mid b_i$ using
  task-specific standard errors (`load_lmarena_pseudo_oracle` already exports
  `bias_se` for this purpose). The standardized $W_1$ effect size remains
  directly descriptive.
- The permutation calibration treats task labels as exchangeable under the
  null. Shared LLMs induce cross-task dependence, so the p-value should be read
  as a diagnostic rather than a full proof of the paper's joint-independence
  assumptions.
- Quartiles are pre-specified for interpretability. Other bin counts should be
  treated as sensitivity analyses, not searched post hoc without recalibration.
- Ties in $\tilde{\theta}_i$ are split deterministically by original task order
  rather than kept together. Bins are fixed across permutations, so this does
  not affect validity of the p-value.

---

## References

- Fisher, R. A. (1935). *The Design of Experiments.* Oliver and Boyd.
- Pitman, E. J. G. (1937). Significance tests which may be applied to samples
  from any populations. *Journal of the Royal Statistical Society, Supplement*,
  4(1), 119–130.
- Phipson, B. and Smyth, G. K. (2010). Permutation p-values should never be
  zero. *Statistical Applications in Genetics and Molecular Biology*, 9(1).
- Villani, C. (2009). *Optimal Transport: Old and New.* Springer.
