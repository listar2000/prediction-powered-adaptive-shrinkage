# Implementation note: empirical-Bayes double shrinkage baseline

This note documents the implementation of Rosenman, Dominici, and Miratrix, **"Empirical Bayes Double Shrinkage for Combining Biased and Unbiased Causal Estimates"** (arXiv:2309.06727) in this codebase. The implementation lives in:

- `src/pas/estimators/double_shrinkage.py` (model fit and point estimators)
- `src/pas/intervals/double_shrinkage_cis.py` (robust EB confidence intervals)
- `src/pas/intervals/robust_eb.py` (robust EBCI critical values)
- registry keys `double_shrinkage_{mm1,mm2,mle,ure}_ci` in `pas.intervals.CORE_CI_METHODS`

The main LM Arena benchmark uses `double_shrinkage_mle_ci`, because the source paper reports the MLE version as particularly competitive while the four variants behave similarly for interval length and average coverage.

## 1. Original model and estimator

For tasks $i = 1, \dots, K$, the source paper writes the unbiased estimator as $u_i = \hat\tau_{u i}$, the possibly biased estimator as $v_i = \hat\tau_{b i}$, the target as $\tau_i$, and the bias as $\xi_i$. Its hierarchical model (Eq. 2) is

$$
\tau_i \sim \mathcal{N}\!\left(0, \eta^2\right), \qquad
\xi_i \sim \mathcal{N}\!\left(0, \gamma^2\right),
$$

$$
u_i \mid \tau_i \sim \mathcal{N}\!\left(\tau_i, s_{ui}^2\right), \qquad
v_i \mid \tau_i, \xi_i \sim \mathcal{N}\!\left(\tau_i + \xi_i, s_{bi}^2\right),
$$

with conditional independence of $u_i$ and $v_i$ and diagonal covariance across tasks.

The posterior mean (Eq. 4) is

$$
\lambda_i = \frac{\gamma^2 + s_{bi}^2}{\gamma^2 + s_{bi}^2 + s_{ui}^2},
\qquad
a_i = \frac{\eta^2 \left(\gamma^2 + s_{bi}^2 + s_{ui}^2\right)}
           {s_{ui}^2 \left(\gamma^2 + s_{bi}^2\right) + \eta^2 \left(\gamma^2 + s_{bi}^2 + s_{ui}^2\right)},
$$

$$
\psi_i\!\left(\gamma^2, \eta^2\right) = a_i \left\{\lambda_i u_i + (1 - \lambda_i)\, v_i\right\}.
$$

This is "double" shrinkage:

1. $\lambda_i u_i + (1-\lambda_i)\,v_i$ combines the unbiased and biased estimators.
2. $a_i$ shrinks that combination toward the prior center, originally zero.

The implementation permits a fixed `center=c` by applying the original method to $u_i - c$ and $v_i - c$, then adding $c$ back. For LM Arena, the benchmark passes `center=0.5`, which is simply the original zero-centered method applied to the centered win-rate estimand $\theta_i - 0.5$.

## 2. Exact correspondence to the empirical-Bayes rebiasing framework

Our paper uses a biased estimator $\hat\theta_i^{b}$, an unbiased estimator $\hat\theta_i^{ub}$ (equivalently a fully debiased estimator), and the implicit bias estimator

$$
\hat b_i = \hat\theta_i^{b} - \hat\theta_i^{ub}.
$$

For the double-shrinkage baseline, make the identification

$$
\theta_i \leftrightarrow \tau_i, \qquad
b_i \leftrightarrow \xi_i, \qquad
\hat\theta_i^{ub} \leftrightarrow u_i, \qquad
\hat\theta_i^{b} \leftrightarrow v_i, \qquad
\mathrm{Var}\!\left(\hat\theta_i^{ub}\right) \leftrightarrow s_{ui}^2, \qquad
\mathrm{Var}\!\left(\hat\theta_i^{b}\right) \leftrightarrow s_{bi}^2.
$$

Conditional independence of $u_i$ and $v_i$ implies, in our paper's primitive $(\hat\theta_i^{b}, \hat b_i)$ parameterization,

$$
\mathrm{Var}\!\left(\hat b_i\right) = s_{ui}^2 + s_{bi}^2,
\qquad
\mathrm{Cov}\!\left(\hat\theta_i^{b}, \hat b_i\right) = s_{bi}^2.
$$

Under the normal bias prior $b_i \sim \mathcal{N}(0, \gamma^2)$, our oracle normal-prior rebiasing estimator becomes

$$
\hat\theta_i^{ub} + \frac{s_{ui}^2}{\gamma^2 + s_{ui}^2 + s_{bi}^2}
\left(\hat\theta_i^{b} - \hat\theta_i^{ub}\right)
= \lambda_i u_i + (1-\lambda_i)\, v_i.
$$

Thus the **first** shrinkage step of Rosenman et al. is exactly normal-prior empirical-Bayes rebiasing under their independent-estimator model. Their distinctive additional assumption is the proper prior on the target, $\theta_i - c \sim \mathcal{N}(0, \eta^2)$, which yields the second factor $a_i$.

This difference matters inferentially:

- **Our method:** partially Bayes; only nuisance biases are exchangeable and the targets $\theta_i$ remain fixed.
- **Double shrinkage:** also pools the targets toward a fixed center. Its robust intervals target average/empirical-Bayes coverage under target and bias second-moment restrictions, rather than the fixed-target partially-Bayes guarantee in our paper.

## 3. Mapping used for LM Arena

A faithful application of the source paper requires two conditionally independent estimators. For each LLM pair:

$$
\begin{aligned}
u_i &= \text{mean of labelled human preferences}, \\
v_i &= \text{mean of reward-model predictions on the disjoint unlabelled sample}, \\
s_{ui}^2 &= \widehat{\mathrm{Var}}\!\left(Y_{\mathrm{labelled}}\right) / n_i, \\
s_{bi}^2 &= \widehat{\mathrm{Var}}\!\left(\mathrm{pred}_{\mathrm{unlabelled}}\right) / N_i.
\end{aligned}
$$

This is implemented by `independent_estimator_components(data)`.

We intentionally do **not** feed the PT estimator into the baseline. PT reuses both labelled predictions and the unlabelled prediction mean and is correlated with $v_i$; inserting it into the original independent-estimator formulas would no longer reproduce the method in the cited paper. A correlated-estimator generalization is possible, but it would be a new method rather than a faithful baseline.

## 4. Hyperparameter estimators

All four source-paper variants are implemented.

### MM1

$$
\hat\eta^2 = \Big[K^{-1} \sum_i \big(u_i^2 - s_{ui}^2\big)\Big]_+,
\qquad
\hat\gamma^2 = \Big[K^{-1} \sum_i \big\{(u_i - v_i)^2 - s_{ui}^2 - s_{bi}^2\big\}\Big]_+.
$$

### MM2

$$
\hat\eta^2 = \Big[K^{-1} \sum_i \big(u_i^2 - s_{ui}^2\big)\Big]_+,
\qquad
\hat\gamma^2 = \Big[K^{-1} \sum_i \big\{v_i^2 - u_i^2 + s_{ui}^2 - s_{bi}^2\big\}\Big]_+.
$$

### Marginal MLE

The implementation minimizes the negative of the factorized marginal log-likelihood stated in Eqs. 5–6 of the source paper:

$$
\frac{1}{2} \sum_i \left[
\log\big(\eta^2 + s_{ui}^2\big) + \frac{u_i^2}{\eta^2 + s_{ui}^2}
+ \log\big(\eta^2 + \gamma^2 + s_{bi}^2\big) + \frac{v_i^2}{\eta^2 + \gamma^2 + s_{bi}^2}
\right],
$$

subject to $\eta^2, \gamma^2 \ge 0$. We use bounded multi-start L-BFGS-B and retain the best finite objective.

**Faithfulness note.** Under the literal joint hierarchy, $u_i$ and $v_i$ share $\tau_i$ and therefore have marginal covariance $\eta^2$. The manuscript nevertheless writes the product of their two univariate marginals in its MLE section. The code reproduces that displayed objective exactly (equivalently, it can be read as a composite marginal likelihood); it does not silently replace it by a correlated bivariate-normal likelihood.

### URE

For fixed $(\eta^2, \gamma^2)$, let $\psi_i$, $a_i$, and $\lambda_i$ be as above. Eq. 8 gives

$$
\mathrm{URE}\!\left(\eta^2, \gamma^2\right)
= \sum_i s_{ui}^2
+ \sum_i \left(\psi_i - u_i\right)^2
- 2 \sum_i s_{ui}^2 \left(1 - a_i \lambda_i\right).
$$

The implementation minimizes this over the nonnegative orthant using the same bounded multi-start strategy.

## 5. Robust empirical-Bayes interval

The source paper's Definition 1 uses the robust critical value of Armstrong, Kolesár, and Plagborg-Møller. In a form that is easiest to audit in code, define

$$
\mathrm{sv}_i
= a_i^2 \left\{\lambda_i^2 s_{ui}^2 + (1-\lambda_i)^2 s_{bi}^2\right\},
\qquad
m_{2,i}
= \frac{(a_i - 1)^2\, \eta^2 + \left\{a_i (1-\lambda_i)\right\}^2 \gamma^2}{\mathrm{sv}_i},
$$

stored as `sampling_variance` and `normalized_bias_m2` on the fitted object. The robust interval is

$$
\psi_i \pm \mathrm{cva}_\alpha\!\left(m_{2,i}\right) \sqrt{\mathrm{sv}_i},
$$

where $\mathrm{cva}_\alpha(m_2)$ is the worst-case two-sided critical value over distributions of normalized conditional bias having second moment at most $m_2$. Algebraically, $m_{2,i}$ equals the paper's displayed $c_i$:

$$
c_i = \frac{s_{ui}^2 \left[\gamma^2\big(\eta^2 + 2 s_{bi}^2\big) + \gamma^4 + s_{bi}^4\right]}
           {\eta^2 \left[\big(\gamma^2 + s_{bi}^2\big)^2 + s_{bi}^2 s_{ui}^2\right]}.
$$

`src/pas/intervals/robust_eb.py` ports the exact $\kappa = \infty$ scalar solver from the public `ebci` implementation. Calling it hundreds of times inside every Monte Carlo replicate is unnecessarily expensive, so the benchmark default uses a packaged exact lookup table for $\alpha \in \{0.01, 0.05, 0.10, 0.20, 0.30\}$ and rounds $m_2$ upward to the next grid point. This makes the lookup conservative relative to the tabulated exact values; the packaged grid is dense enough (2000 log-spaced points) that the resulting width inflation is below 1% throughout. Set `cv_mode="exact"` to use root finding task by task. The table is regenerated by `src/scripts/generate_robust_eb_table.py`.

The cited baseline derives only symmetric two-sided intervals. The implementation therefore rejects one-sided alternatives instead of silently substituting an unsupported formula.

## 6. Finite-sample truncation: explicit implementation choice

The source paper notes that estimates of $\eta^2$ and $\gamma^2$ may hit zero and says it applies a truncation analogous to Morris/Armstrong et al., but it does not give the exact numerical rule. This is the only material implementation detail that cannot be reconstructed uniquely from the manuscript.

The code exposes both choices:

- `finite_sample_correction="none"`: use the equations exactly; the estimator and interval can collapse when $\hat\eta^2 = 0$.
- `finite_sample_correction="pmt"` (benchmark default): apply the equal-weight posterior mean truncation (PMT) floor used by `ebci` to both latent second moments. For measurement variances $q_i$, the floor is

$$
\frac{2 \, K^{-1} \sum_i q_i^2}{K \cdot K^{-1} \sum_i q_i}
= \frac{2 \, \overline{q^2}}{K \, \overline{q}}.
$$

  For $\eta^2$, $q_i = s_{ui}^2$. For $\gamma^2$, the observable $v_i - u_i$ has measurement variance $q_i = s_{ui}^2 + s_{bi}^2$ under the baseline's independence model.

This choice is transparent and switchable; it should not be described as a verbatim recovery of unpublished source code.

## 7. Code integration and benchmark command

`pas.intervals.CORE_CI_METHODS` registers:

```text
double_shrinkage_mm1_ci
double_shrinkage_mm2_ci
double_shrinkage_mle_ci
double_shrinkage_ure_ci
```

The standard LM Arena runner includes the MLE variant and supplies the fixed win-rate center:

```python
ci_kwargs = {
    "double_shrinkage_mle_ci": {
        "center": 0.5,
        "finite_sample_correction": "pmt",
        "cv_mode": "lookup",
    },
}
```

Run the same benchmark pipeline as before:

```bash
uv run python src/scripts/run_lmarena_ci.py --trials 200 --num-workers 4
```

No NPMLE/MOSEK dependency is needed for the double-shrinkage baseline itself.

## 8. Validation performed

The port was checked numerically against the source paper and the reference `ebci` implementation:

- $\lambda_i$, $a_i$, and $\psi_i$ agree with the raw posterior-mean expression (Eq. 3), and the first shrinkage step reproduces the normal-prior rebiasing identity of Section 2;
- the general normalized-bias expression $m_{2,i}$ agrees with the paper's closed-form $c_i$ to numerical precision;
- MM2 and MLE coincide under homoscedasticity, as claimed in Section 2.3 of the source paper;
- at the solved critical value, an independent linear program over discretized bias distributions recovers worst-case noncoverage $\alpha$;
- lookup-mode critical values are conservative relative to the exact solver, with sub-1% width inflation on the packaged grid;
- the fitted rule is translation-equivariant in the fixed center;
- under the paper's own hierarchical model, the intervals attain their nominal empirical-Bayes coverage with a modest degree of overcoverage, matching the behavior reported in the source paper.
