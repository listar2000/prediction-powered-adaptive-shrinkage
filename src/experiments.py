"""
Common experiment utilities for running benchmarks and visualizations across different datasets.
"""
from datasets.dataset import PasDataset
from estimators import ALL_ESTIMATORS
from utils import get_mse, get_default_args, get_decrease_fraction
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import json


BENCHMARK_FOLDER = Path(__file__).parent / "benchmarks"

def run_benchmark(dataset: PasDataset,
                 trials: int = 100,
                 summary: bool = True,
                 save_results: bool = True,
                 estimators: Optional[dict] = None,
                 estimator_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """ Run benchmark experiments on any dataset.
    
    Args:
        dataset: Dataset to run experiments on
        trials: Number of trials to run
        summary: Whether to print summary statistics
        estimators: Dictionary of estimators to use. If None, uses ALL_ESTIMATORS
        estimator_kwargs: Dictionary of keyword arguments to pass to each estimator.
            Each key is an estimator name, and the value is a dictionary of arguments
            to pass to that estimator.

    Example estimator_kwargs:
        {
            "eb_sure": {
                "hetero_var_y": True,
                "cutoff": 0.99
            }
        }
    
    Returns:
        DataFrame containing MSE results for each estimator across trials
    """
    estimators = estimators or ALL_ESTIMATORS
    col_names = list(estimators.keys()) + [estimator + "_frac" for estimator in estimators.keys()]
    mse_results = pd.DataFrame(columns=col_names)
    base_seed = dataset.split_seed
    for i in tqdm(range(trials)):
        # randomize the dataset (train-test split)
        dataset.reload_data(split_seed=i + base_seed)
        true_theta = dataset.true_theta

        mle = None
        for estimator_name, estimator_func in estimators.items():
            kwargs = estimator_kwargs.get(estimator_name, {}) if estimator_kwargs else {}
            theta_hat = estimator_func(dataset, **kwargs)
            if estimator_name == "mle":
                mle = theta_hat
            mse = get_mse(theta_hat, true_theta)
            mse_results.loc[i, estimator_name] = mse
            frac = get_decrease_fraction(true_theta, theta_hat, mle)
            mse_results.loc[i, estimator_name + "_frac"] = frac

    summary_text = [f"\nResults for {dataset.dataset_name}:\n",
                    f"Mean of metrics:\n {mse_results.mean().to_string(float_format='{:.8f}'.format)}",
                    f"SD of metrics:\n {mse_results.std().to_string(float_format='{:.8f}'.format)}",
                    f"SE of mean metrics:\n {mse_results.sem().to_string(float_format='{:.8f}'.format)}"]

    if summary:
        print("\n".join(summary_text))

    if save_results:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        save_dir = BENCHMARK_FOLDER / f"{dataset.dataset_name}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        mse_results.to_csv(save_dir / "results.csv")
        
        estimator_args = {}
        for name, func in estimators.items():
            default_args = get_default_args(func)
            if estimator_kwargs and name in estimator_kwargs:
                default_args.update(estimator_kwargs[name])
            estimator_args[name] = default_args
        
        with open(save_dir / "estimator_args.json", 'w') as f:
            json.dump(estimator_args, f, indent=2)

        # save the summary text
        with open(save_dir / "summary.txt", 'w') as f:
            f.write("\n".join(summary_text))

    return mse_results


def visualize_benchmark(mse_results: pd.DataFrame,
                       dataset_name: str,
                       output_path: Optional[Path] = None):
    """ Visualize benchmark results for any dataset.
    
    Args:
        mse_results: DataFrame containing MSE results
        dataset_name: Name of dataset for plot title
        output_path: Path to save plot. If None, plot is displayed but not saved
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=mse_results, orient="v")
    plt.xlabel("MSE")
    plt.title(f"Benchmark results on {dataset_name}")
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def run_mixed_estimator_experiment(dataset: PasDataset,
                                 estimator1_name: str,
                                 estimator2_name: str,
                                 trials: int = 100,
                                 num_weights: int = 50,
                                 output_path: Optional[Path] = None):
    """ Run experiment mixing two estimators with different weights.
    
    Args:
        dataset: Dataset to run experiments on
        estimator1_name: Name of first estimator in ALL_ESTIMATORS
        estimator2_name: Name of second estimator in ALL_ESTIMATORS
        trials: Number of trials to run
        num_weights: Number of weight values to try between 0 and 1
        output_path: Path to save plot. If None, plot is displayed but not saved
    """
    assert estimator1_name in ALL_ESTIMATORS and estimator2_name in ALL_ESTIMATORS, \
        f"Estimators {estimator1_name} and {estimator2_name} must be in ALL_ESTIMATORS"

    weights = np.linspace(0, 1, num_weights)
    mse_results = np.zeros((trials, num_weights))

    for i in tqdm(range(trials)):
        dataset.reload_data(split_seed=i + 12345)
        true_theta = dataset.true_theta

        estimates1 = ALL_ESTIMATORS[estimator1_name](dataset)
        estimates2 = ALL_ESTIMATORS[estimator2_name](dataset)

        for j, w in enumerate(weights):
            mixed_estimates = w * estimates1 + (1 - w) * estimates2
            mse = get_mse(mixed_estimates, true_theta)
            mse_results[i, j] = mse

    # Plot results
    plt.figure(figsize=(10, 6))
    mean_mse = mse_results.mean(axis=0)
    sd_mse = mse_results.std(axis=0)
    plt.plot(weights, mean_mse, label="Mean MSE")
    plt.fill_between(weights, mean_mse - sd_mse, mean_mse + sd_mse,
                     alpha=0.2, label="Mean MSE +/- 1 SD")
    plt.xlabel(f"Weight for {estimator1_name}")
    plt.ylabel("MSE")
    plt.title(f"Mixed {estimator1_name} and {estimator2_name} loss")
    plt.legend()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
