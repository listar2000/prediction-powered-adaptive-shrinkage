# Implementation note: robust empirical-Bayes confidence intervals

Reference: Timothy B. Armstrong, Michal Kolesár, and Mikkel Plagborg-Møller,
"Robust Empirical Bayes Confidence Intervals," *Econometrica* 90(6):2567–2602,
2022, and the authors' MIT-licensed R package
[`kolesarm/ebci`](https://github.com/kolesarm/ebci).

This baseline is the *single*-shrinkage counterpart to the double-shrinkage
baseline documented in `double_shrinkage_implementation_note.md`. It does not
use a biased prediction-only estimator at all: it starts from an unbiased
estimator of each task mean and shrinks it toward a weighted regression fit,
then widens the interval by a critical value that is valid for *any* bias
distribution obeying an estimated second- (and fourth-) moment bound.

---

## 1. Original method

Given unbiased preliminary estimators $Y_i \sim N(\theta_i, s_i^2)$ with known
$s_i$, and a design matrix $X$ (the R formula's right-hand side), the method
fits a weighted least-squares line $\mu_i = X_i'\delta$ and shrinks toward it:

$$
\hat{\theta}_i = \mu_i + w_i (Y_i - \mu_i),
\qquad
w_i = \frac{\mu_2}{\mu_2 + s_i^2},
$$

where $\mu_2$ is the second moment of the residual $\theta_i - X_i'\delta$.
The robust EBCI is

$$
\hat{\theta}_i \pm \mathrm{cva}_\alpha\!\left(m_{2,i}, \kappa\right) \cdot w_i s_i,
\qquad
m_{2,i} = \frac{s_i^2}{\mu_2},
$$

with $\kappa$ the kurtosis of $\theta_i - X_i'\delta$.

The identity $m_{2,i} = s_i^2/\mu_2$ is worth spelling out, because it is the
one step where the port could silently diverge. The conditional bias of
$\hat{\theta}_i$ is $(1-w_i)(\mu_i - \theta_i)$; normalizing by the standard
deviation $w_i s_i$ of the estimator gives a normalized bias whose second
moment is

$$
\frac{(1-w_i)^2 \mu_2}{w_i^2 s_i^2}
= \frac{s_i^4 \mu_2 / (\mu_2+s_i^2)^2}{\mu_2^2 s_i^2 / (\mu_2+s_i^2)^2}
= \frac{s_i^2}{\mu_2}.
$$

Kurtosis is scale-invariant, so the same $\kappa$ applies to every task.

$\mathrm{cva}_\alpha(m_2, \kappa)$ is the smallest $\chi$ such that the
worst-case non-coverage

$$
\rho(m_2, \kappa, \chi)
= \sup_{F} \int r(t, \chi)\, \mathrm{d}F(t),
\qquad
r(t,\chi) = \Phi(-\sqrt{t}-\chi) + \Phi(\sqrt{t}-\chi),
$$

over distributions $F$ of the squared normalized bias satisfying
$\mathbb{E}_F[t] = m_2$ and $\mathbb{E}_F[t^2] \le \kappa m_2^2$, equals
$\alpha$. Coverage is therefore *average* (empirical-Bayes) coverage across
tasks, not coverage for each task separately.

---

## 2. Mapping used for LM Arena

Task $i$ is one model pair; $\theta_i$ is the human win rate. The unbiased
preliminary estimator is the power-tuned PPI estimator that the paper already
reports as the `pt_ci` baseline:

$$
Y_i = \bar{y}_i + \lambda_i\left(\bar{f}_i^{\,\mathrm{unl}} - \bar{f}_i^{\,\mathrm{lab}}\right),
\qquad
s_i^2
= \frac{\widehat{\mathrm{Var}}(\lambda_i f)}{N_i}
+ \frac{\widehat{\mathrm{Var}}(y - \lambda_i f)}{n_i}.
$$

`get_unbiased_estimates_and_ses` reproduces `get_pt_ppi_cis` exactly: the
returned centres and standard errors are bitwise identical to the ones the
`pt_ci` baseline uses. Robust EB therefore shrinks precisely the interval the
PT row of the benchmark table reports, which makes the comparison in the
figure apples-to-apples.

The configuration used in the benchmark corresponds to the R call

```r
ebci(Y ~ 1, se = se, weights = 1 / se^2, alpha = alpha, fs_correction = "PMT")
```

that is, shrink toward the inverse-variance-weighted grand mean
(`shrink_to="grand_mean"`), fit the moments with weights $1/s_i^2$, and apply
the posterior-mean-truncation finite-sample correction.

`base_estimator` also accepts `"ppi"` (vanilla PPI, $\lambda_i \equiv 1$) and
`"classical"` (labelled-sample mean only); both likewise reproduce the
corresponding repository baselines exactly.

---

## 3. Moment estimation and the finite-sample correction

With residuals $\varepsilon_i = Y_i - \mu_i$ and weights $\omega_i$, define

$$
W_{2i} = \varepsilon_i^2 - s_i^2,
\qquad
W_{4i} = \varepsilon_i^4 - 6 s_i^2 \varepsilon_i^2 + 3 s_i^4,
$$

which are unbiased for $\mu_2$ and $\mu_4$. The uncorrected estimates
$\tilde{\mu}_2, \tilde{\mu}_4$ are their $\omega$-weighted means. These can be
negative in finite samples, so `fs_correction="PMT"` floors them at

$$
\mathrm{trim}_2 = \frac{2\,\overline{\omega^2 s^4}}{\left(\sum_i \omega_i\right)\overline{\omega s^2}},
\qquad
\mathrm{trim}_4 = \frac{32\,\overline{\omega^2 s^8}}{\left(\sum_i \omega_i\right)\overline{\omega s^4}},
$$

giving $\mu_2 = \max(\tilde{\mu}_2, \mathrm{trim}_2)$ and
$\kappa = \max\!\left(\tilde{\mu}_4/\mu_2^2,\ 1 + \mathrm{trim}_4/\mu_2^2\right)$.
`"FPLIB"` (truncated-normal mean) and `"none"` are also implemented, matching
the three branches of `ebci:::moments`.

Because $\mathrm{trim}_2 > 0$ whenever some $s_i > 0$, the PMT default cannot
produce $\mu_2 = 0$, so the degenerate "no shrinkage possible" branch is never
reached in the benchmark. It is still handled: if $\mu_2 \le 0$ the interval
falls back to the unshrunk Wald interval with a `RuntimeWarning`.

On LM Arena the fitted values are stable across resamples: $\mu_2 \approx
0.0688$, $\kappa \approx 2.46$, and $m_{2,i} \in [0, 0.404]$.

---

## 4. Critical values

`src/pas/intervals/robust_eb.py` is a line-by-line port of `R/cv.R`:
`r`/`r1`/`r2`/`r3` (the non-coverage function and its first three
derivatives, including the L'Hôpital limits used near $t=0$), `rt0` (tangent
point and inflection point), `rho0` (second-moment-only concave envelope),
`delta`/`delta1`, `lam` (maximization of $\delta$), `rho` (the two-constraint
dual), `CVb`, and `cva` itself.

Two deliberate departures from the R source, both flagged in code:

- `lam` raises an error in R when $\delta$'s derivative pattern is neither
  monotone nor near-zero. The port instead emits a `RuntimeWarning` and falls
  back to a dense deterministic bracket over the same objective, so a single
  pathological task cannot abort a 200-replicate benchmark. The warning makes
  any such occurrence visible; none occurred in the LM Arena runs.
- `rho` in R optionally re-verifies its dual solution against a discretized
  primal linear program (`check=TRUE`). The port omits that self-check. It is
  replaced by an external audit (§6) that solves the primal LP independently.

Interior optimizations additionally evaluate both interval endpoints, which R's
`optimize` does not guarantee; this can only *increase* the computed worst-case
non-coverage and therefore only widens the interval.

### Lookup tables versus exact solving

Two packaged tables exist:
`robust_eb_cv_table.npz` ($\kappa=\infty$) and
`robust_eb_finite_cv_table.npz` (finite $\kappa$, $m_2 \le 2$,
$\kappa \le 10$, for $\alpha \in \{0.01, 0.05, 0.10, 0.20, 0.30\}$).
`cv_mode="lookup"` rounds $m_2$ and $\kappa$ *up*, which is conservative
because $\mathrm{cva}$ is nondecreasing in both. Values outside a table fall
back to the exact solver.

**The LM Arena benchmark uses `cv_mode="exact"`.** At the $(m_2, \kappa)$
values this dataset actually produces, the finite-$\kappa$ lookup grid would
inflate critical values by about 1.0% on average and up to 2.4%. Since robust
EB is a *competitor* baseline, systematically widening it would flatter our own
methods, so we pay the extra runtime (roughly 12 s per replicate across all
five $\alpha$ values) and solve `cva` per task exactly as `ebci()` does. The
double-shrinkage baseline was switched to exact mode for the same reason.

Regenerate the finite-$\kappa$ table with:

```bash
uv run python src/scripts/generate_robust_eb_finite_table.py
```

---

## 5. Code integration

| Piece | Location |
| --- | --- |
| Model fit, moment estimation, PAS adapters | `src/pas/estimators/robust_eb.py` |
| Interval construction | `src/pas/intervals/robust_eb_cis.py` |
| `cva` critical values (ported from `ebci`) | `src/pas/intervals/robust_eb.py` |
| Finite-$\kappa$ table generator | `src/scripts/generate_robust_eb_finite_table.py` |
| Registry key | `robust_eb_ci` in `pas.intervals.CORE_CI_METHODS` |
| Estimator registry key | `robust_eb` in `pas.estimators.ALL_ESTIMATORS` |
| Tests | `tests/test_robust_eb.py` |

Registration lives in `pas/intervals/__init__.py` rather than a side module
because `pas.experiments._ci_run_chunk` re-imports `CORE_CI_METHODS` inside
spawned worker processes.

Benchmark command:

```bash
MOSEKLM_LICENSE_FILE=$PWD/tmp/mosek.lic \
  uv run python src/scripts/run_lmarena_ci.py \
    --trials 200 --num-workers 6 \
    --csv data/lmarena/clean_data/clean_summary_v2.csv \
    --out-dir tmp/lmarena_ci_results
```

---

## 6. Validation performed

Reference values and formulas were checked against the upstream R package,
not against a paraphrase of it.

1. **Upstream reference critical values.** All four `cva` values asserted in
   `ebci`'s own `tests/testthat/test-cv.R` reproduce to $2\times10^{-8}$
   relative error: $\mathrm{cva}(25,3)=11.88358367$,
   $\mathrm{cva}(1,3;\alpha{=}0.2)=1.8494683$,
   $\mathrm{cva}(100,3)=24.86172217$,
   $\mathrm{cva}(7.84,8;\alpha{=}0.1)=7.0779747158$. The package's other
   `test-cv.R` assertions also hold: the $\mathrm{cva}(0,\cdot)=z_{1-\alpha/2}$
   and $\mathrm{cva}(m_2,1)=\mathrm{CVb}(\sqrt{m_2})$ identities, the
   non-binding-$\kappa$ equalities at $(m_2,\kappa)=(1,10^4)$ and $(4,20)$, the
   strict inequality at $(0.25,15)$, Chebyshev sharpness as
   $m_2\to\infty$, $\mathrm{lam}(0, 42.2201).x_0 = 1932.78377442$,
   $\rho(1,3,11.9699639845)=6.0216\times10^{-5}$, and
   $\rho(7.84,8,7.1324290148)=0.09805099$.
2. **Independent primal verification.** For six $(m_2,\kappa,\alpha)$
   combinations, the worst-case non-coverage was recomputed by maximizing
   $\sum_j p_j r(x_j,\chi)$ subject to $\sum_j p_j = 1$,
   $\sum_j p_j x_j = m_2$, $\sum_j p_j x_j^2 \le \kappa m_2^2$ over a 40,000
   point grid using `scipy.optimize.linprog`. At the solved critical value this
   equals $\alpha$ to within $5\times10^{-4}$ in every case. This shares no
   code with the dual solver being tested.
3. **Moment estimation.** `_moment_estimates` matches an independent
   transcription of `ebci:::moments` to $10^{-10}$ relative error over 300
   random draws, for all three `fs_correction` settings.
4. **End-to-end.** The full pipeline (weighted regression, moments,
   shrinkage, critical values, half-lengths) matches an independent
   transcription of `ebci()` on the `y`/`se` vectors from `ebci`'s
   `test-eb.R`, for `Y ~ 1` and `Y ~ 0` and $\alpha \in \{0.05, 0.10, 0.20\}$.
5. **PAS adapters.** Centres and standard errors from
   `get_unbiased_estimates_and_ses` are bitwise identical to `get_pt_ppi_cis`
   (`base_estimator="pt"`) and `get_mle_cis` (`base_estimator="classical"`).
6. **Monotonicity and table integrity.** `cva` is nondecreasing in both $m_2$
   and $\kappa$ at all five benchmark $\alpha$ values; every entry of the
   packaged finite-$\kappa$ table equals the exact solver to $10^{-13}$; and
   over 600 random in-range probes the lookup never returned a value below the
   exact one.

### Known divergence

Under `fs_correction="none"` *and* $\tilde{\mu}_2 \le 0$ *and*
$\tilde{\mu}_4 < 0$, R returns $\kappa = 1$ (from `max(-Inf, 1)`) whereas the
port returns $\kappa = \infty$. This branch is unreachable in every configured
PAS path: PMT is the default and cannot yield $\mu_2 = 0$, and when
$\mu_2 = 0$ the interval code short-circuits to the unshrunk Wald interval
without consulting $\kappa$. Returning $\infty$ is also the conservative
choice of the two.
