from datasets.dataset import PasDataset
import numpy as np
import numpy.random as npr
from typing import Tuple, List


class GaussianSyntheticDataset(PasDataset):
    def __init__(self, good_f: bool = True, M: int = 100, split_seed: int = 42, verbose: bool = False, has_true_vars: bool = True):
        self.M = M
        self.ns = np.repeat(20, self.M)
        self.Ns = np.repeat(80, self.M)
        self.split_seed = split_seed
        self.additional_y_variance = 0.05
        self.sigma_x = 0.1
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
            x_y_mean = np.array([mu_x_s[i], mu_y_s[i]])

            adjusted_sigma_y = np.sqrt(
                (slopes[i] * self.sigma_x)**2 + self.additional_y_variance)
            # Keep regression slope intact
            adjusted_cov_xy = slopes[i] * self.sigma_x**2

            x_y_cov = np.array([[self.sigma_x**2, adjusted_cov_xy],
                                [adjusted_cov_xy, adjusted_sigma_y**2]])

            # using Monte-Carlo to calculate the true variances & covariancse (if `has_true_vars` is `True`)
            DEBUG_FLAG = True
            run_monte_carlo = self.has_true_vars and (not self.good_f or DEBUG_FLAG)
            total_size = (self.ns[i] + self.Ns[i]) * 500 if run_monte_carlo else (self.ns[i] + self.Ns[i])
            x_y = npr.multivariate_normal(x_y_mean, x_y_cov, total_size).T
            # apply predictions
            total_preds = self.pred_f(x_y[0, :])

            # calculate the `true` covariance between the ys and the predicted values
            if not self.good_f or DEBUG_FLAG:
                cov_mat = np.cov(total_preds, x_y[1, :], ddof=1)
                true_vars.append(cov_mat[0, 0])
                true_covs.append(cov_mat[0, 1])
                true_y_vars.append(cov_mat[1, 1])
            else:
                k = 4 * mu_x_s[i] ** 2 * self.sigma_x ** 2
                true_y_vars.append(k + self.additional_y_variance)
                true_vars.append(k + 2 * self.sigma_x ** 4)
                true_covs.append(k)

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
