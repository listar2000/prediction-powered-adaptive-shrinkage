"""
Trivial estimators
"""
import numpy as np
from datasets.dataset import PasDataset


def get_mle_estimators(data: PasDataset) -> np.ndarray:
    """ Obtain the MLE estimator, i.e. the mean response for each problem.
    """
    return np.array([y.mean() for y in data.y_labelled])


def get_pred_mean_estimators(data: PasDataset) -> np.ndarray:
    """ Obtain the prediction mean estimator for each problem.
    """
    return np.array([pred.mean() for pred in data.pred_unlabelled])
