from pas.datasets.dataset import PasDataset
import math
import numpy as np
import numpy.random as npr
from typing import Tuple, List


class GaussianSyntheticDataset(PasDataset):
    #: Source of the "true" second moments when `has_true_vars=True`.
    #: `True` (default): estimate them by Monte Carlo over (n_j+N_j)*500 draws
    #: per problem -- the historical benchmark path. `False`:
    #: use the closed forms of Appendix E.1 (for f(x)=|x| with the corrected
    #: gamma_j = 2 eta_j psi^2 (2 Phi(eta_j/psi) - 1); the camera-ready prints
    #: a wrong expression) and draw only the n_j+N_j data points. Toggling
    #: changes how much of the RNG stream each problem consumes, so the two
    #: settings produce different (identically distributed) data realizations.
    DEBUG_FLAG = True

    def __init__(
        self,
        good_f: bool = True,
        M: int = 100,
        split_seed: int = 42,
        verbose: bool = False,
        has_true_vars: bool = True,
        sigma_x: float = 0.1,
    ):
        if sigma_x <= 0:
            raise ValueError("sigma_x must be positive")
        self.M = M
        self.ns = np.repeat(20, self.M)
        self.Ns = np.repeat(80, self.M)
        self.split_seed = split_seed
        self.additional_y_variance = 0.05
        self.sigma_x = float(sigma_x)
        self.good_f = good_f

        self.mean_y_f = lambda x: x ** 2

        if good_f:
            self.pred_f = self.mean_y_f
        else:
            self.pred_f = lambda x: np.abs(x)

        super().__init__(
            f"Gaussian_synthetic_{'good' if good_f else 'bad'}", verbose=verbose, has_true_vars=has_true_vars)

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
        npr.seed(self.split_seed)
        mu_x_s = npr.uniform(-1, 1, self.M)
        self.mu_x_s = mu_x_s
        mu_y_s = self.mean_y_f(mu_x_s)

        slopes = 2 * mu_x_s

        pred_labelled, y_labelled, pred_unlabelled, y_unlabelled = [], [], [], []

        true_covs, true_vars, true_y_vars = [], [], []

        for i in range(self.M):
            # using Monte-Carlo to calculate the true variances & covariancse (if `has_true_vars` is `True`)
            run_monte_carlo = self.has_true_vars and self.DEBUG_FLAG
            total_size = (self.ns[i] + self.Ns[i]) * 500 if run_monte_carlo else (self.ns[i] + self.Ns[i])
            if run_monte_carlo:
                x_y_mean = np.array([mu_x_s[i], mu_y_s[i]])
                adjusted_sigma_y = np.sqrt(
                    (slopes[i] * self.sigma_x) ** 2
                    + self.additional_y_variance
                )
                adjusted_cov_xy = slopes[i] * self.sigma_x**2
                x_y_cov = np.array([
                    [self.sigma_x**2, adjusted_cov_xy],
                    [adjusted_cov_xy, adjusted_sigma_y**2],
                ])
                x_y = npr.multivariate_normal(
                    x_y_mean, x_y_cov, total_size
                ).T
            else:
                # Conditional form used by the paper reproduction scripts.
                # It is the same bivariate Gaussian, but fixes the RNG order:
                # first X, then the independent response noise.
                x = npr.normal(mu_x_s[i], self.sigma_x, total_size)
                epsilon = npr.normal(
                    0, np.sqrt(self.additional_y_variance), total_size
                )
                y = (
                    mu_y_s[i]
                    + slopes[i] * (x - mu_x_s[i])
                    + epsilon
                )
                x_y = np.vstack([x, y])
            # apply predictions
            total_preds = self.pred_f(x_y[0, :])

            # calculate the `true` covariance between the ys and the predicted values
            if run_monte_carlo:
                cov_mat = np.cov(total_preds, x_y[1, :], ddof=1)
                true_vars.append(cov_mat[0, 0])
                true_covs.append(cov_mat[0, 1])
                true_y_vars.append(cov_mat[1, 1])
            elif self.has_true_vars:
                # Closed forms of Appendix E.1. Var(Y) = 4 eta^2 psi^2 + c for
                # both predictors.
                eta, psi = mu_x_s[i], self.sigma_x
                k = 4 * eta ** 2 * psi ** 2
                true_y_vars.append(k + self.additional_y_variance)
                if self.good_f:
                    # f(x) = x^2: Var(f) = 4 eta^2 psi^2 + 2 psi^4, Cov(f, Y) = k.
                    true_vars.append(k + 2 * psi ** 4)
                    true_covs.append(k)
                else:
                    # f(x) = |x|: folded-normal moments, a = eta/psi.
                    #   E|X|      = psi sqrt(2/pi) e^{-a^2/2} + eta (2 Phi(a) - 1)
                    #   Var(f)    = eta^2 + psi^2 - (E|X|)^2
                    #   Cov(f, Y) = 2 eta Cov(|X|, X) = 2 eta psi^2 (2 Phi(a) - 1)
                    # via Stein's lemma, Cov(|X|, X) = psi^2 E[sign(X)]. Note the
                    # camera-ready E.1 prints gamma with a Gaussian-pdf factor
                    # instead of (2 Phi(a) - 1); that expression fails numerical
                    # validation (see tests/test_synthetic_moments.py).
                    a = eta / psi
                    phi_cdf = 0.5 * (1.0 + math.erf(a / math.sqrt(2.0)))
                    mean_abs = psi * math.sqrt(2.0 / math.pi) * math.exp(-a ** 2 / 2.0) \
                        + eta * (2.0 * phi_cdf - 1.0)
                    true_vars.append(eta ** 2 + psi ** 2 - mean_abs ** 2)
                    true_covs.append(2.0 * eta * psi ** 2 * (2.0 * phi_cdf - 1.0))

            pred_labelled.append(total_preds[:self.ns[i]])
            y_labelled.append(x_y[1, :self.ns[i]])
            pred_unlabelled.append(
                total_preds[self.ns[i]:(self.ns[i] + self.Ns[i])])
            y_unlabelled.append(x_y[1, self.ns[i]:(self.ns[i] + self.Ns[i])])

        # provide the true vars and covs
        if self.has_true_vars:
            self.true_vars = true_vars
            self.true_covs = true_covs
            self.true_y_vars = true_y_vars

        return pred_labelled, y_labelled, pred_unlabelled, y_unlabelled, mu_y_s

    def reload_data(self, split_seed: int = 42) -> None:
        self.split_seed = split_seed
        pred_labelled, y_labelled, pred_unlabelled, y_unlabelled, mu_y_s = self.load_data()
        self.set_metadata(pred_labelled, y_labelled,
                          pred_unlabelled, y_unlabelled, mu_y_s)
