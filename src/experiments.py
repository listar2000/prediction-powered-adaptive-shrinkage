"""
Common experiment utilities for running benchmarks and visualizations across different datasets.
"""
from datasets.dataset import PasDataset
from estimators import CORE_ESTIMATORS
from utils import get_mse, get_default_args, get_decrease_fraction, get_coverage, get_avg_ci_width
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import json


BENCHMARK_FOLDER = Path(__file__).parent.parent / "results"


def run_benchmark(dataset: PasDataset,
                  trials: int = 100,
                  summary: bool = True,
                  save_results: bool = False,
                  estimators: Optional[dict] = None,
                  estimator_kwargs: Optional[dict] = None,
                  verbose: bool = True) -> pd.DataFrame:
    """ Run benchmark experiments on any dataset.

    Args:
        dataset: Dataset to run experiments on
        trials: Number of trials to run
        summary: Whether to print summary statistics
        estimators: Dictionary of estimators to use. If None, uses CORE_ESTIMATORS
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
    estimators = estimators or CORE_ESTIMATORS
    col_names = list(estimators.keys()) + \
        [estimator + "_frac" for estimator in estimators.keys()]
    mse_results = pd.DataFrame(columns=col_names)
    base_seed = dataset.split_seed
    for i in tqdm(range(trials), disable=not verbose):
        # randomize the dataset (train-test split)
        dataset.reload_data(split_seed=i + base_seed)
        true_theta = dataset.true_theta

        mle = None
        for estimator_name, estimator_func in estimators.items():
            kwargs = estimator_kwargs.get(
                estimator_name, {}) if estimator_kwargs else {}
            theta_hat = estimator_func(dataset, **kwargs)
            if estimator_name == "mle":
                mle = theta_hat
            mse = get_mse(theta_hat, true_theta)
            mse_results.loc[i, estimator_name] = mse
            frac = get_decrease_fraction(true_theta, theta_hat, mle)
            mse_results.loc[i, estimator_name + "_frac"] = frac

    summary_text = [f"\nResults for {dataset.dataset_name}:\n",
                    f"Mean of metrics:\n{mse_results.mean().to_string(float_format='{:.8f}'.format)}",
                    f"SD of metrics:\n{mse_results.std().to_string(float_format='{:.8f}'.format)}",
                    f"SE of mean metrics:\n{mse_results.sem().to_string(float_format='{:.8f}'.format)}"]

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


def run_benchmark_timing(dataset: PasDataset,
                         trials_per_batch: int = 10,
                         num_batches: int = 50,
                         warmup_batches: int = 3,
                         summary: bool = True,
                         save_results: bool = True,
                         estimators: Optional[dict] = None,
                         estimator_kwargs: Optional[dict] = None,
                         verbose: bool = True) -> pd.DataFrame:
    """ Run benchmark to measure computational time for each estimator.

    This function focuses solely on measuring the execution time of each estimator
    without calculating MSE or other accuracy metrics. It measures time for batches
    of trials rather than individual trials to provide more meaningful measurements.

    Args:
        dataset: Dataset to run experiments on
        trials_per_batch: Number of trials to run in each batch
        num_batches: Number of batches to run for timing statistics
        warmup_batches: Number of warmup batches to run before timing (not included in results)
        summary: Whether to print summary statistics
        save_results: Whether to save results to disk
        estimators: Dictionary of estimators to use. If None, uses CORE_ESTIMATORS
        estimator_kwargs: Dictionary of keyword arguments to pass to each estimator.
            Each key is an estimator name, and the value is a dictionary of arguments
            to pass to that estimator.

    Returns:
        DataFrame containing timing results (in seconds) for each estimator across batches
    """
    import time

    estimators = estimators or CORE_ESTIMATORS
    timing_results = pd.DataFrame(columns=list(estimators.keys()))
    base_seed = dataset.split_seed

    # Perform warmup batches first (results not recorded)
    print(
        f"Running {warmup_batches} warmup batches (each with {trials_per_batch} trials)...")
    for batch in range(warmup_batches):
        for estimator_name, estimator_func in estimators.items():
            kwargs = estimator_kwargs.get(
                estimator_name, {}) if estimator_kwargs else {}
            for i in range(trials_per_batch):
                seed = base_seed + batch * trials_per_batch + i
                dataset.reload_data(split_seed=seed)
                _ = estimator_func(dataset, **kwargs)

    # Now run the actual timing batches
    print(
        f"Running {num_batches} timing batches (each with {trials_per_batch} trials)...")
    for batch in tqdm(range(num_batches), disable=not verbose):
        for estimator_name, estimator_func in estimators.items():
            kwargs = estimator_kwargs.get(
                estimator_name, {}) if estimator_kwargs else {}

            # Measure execution time for the entire batch
            start_time = time.time()
            for i in range(trials_per_batch):
                seed = base_seed + (warmup_batches + batch) * \
                    trials_per_batch + i
                dataset.reload_data(split_seed=seed)
                _ = estimator_func(dataset, **kwargs)
            end_time = time.time()

            # Record execution time in seconds for the entire batch
            timing_results.loc[batch, estimator_name] = end_time - start_time

    # Calculate per-trial timing for better interpretability
    per_trial_timing = timing_results / trials_per_batch

    summary_text = [
        f"\nTiming Results for {dataset.dataset_name}:\n",
        f"Batch size: {trials_per_batch} trials per batch\n",
        f"Mean batch execution time (seconds):\n {timing_results.mean().to_string(float_format='{:.6f}'.format)}",
        f"SD of batch execution time (seconds):\n {timing_results.std().to_string(float_format='{:.6f}'.format)}",
        f"Min batch execution time (seconds):\n {timing_results.min().to_string(float_format='{:.6f}'.format)}",
        f"Max batch execution time (seconds):\n {timing_results.max().to_string(float_format='{:.6f}'.format)}",
        f"\nMean per-trial execution time (seconds):\n {per_trial_timing.mean().to_string(float_format='{:.6f}'.format)}",
        f"SD of per-trial execution time (seconds):\n {per_trial_timing.std().to_string(float_format='{:.6f}'.format)}"
    ]

    if summary:
        print("\n".join(summary_text))

    if save_results:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        save_dir = BENCHMARK_FOLDER / \
            f"{dataset.dataset_name}_timing_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save both batch and per-trial timing results
        timing_results.to_csv(save_dir / "batch_timing_results.csv")
        per_trial_timing.to_csv(save_dir / "per_trial_timing_results.csv")

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

    return timing_results


def run_ci_benchmark(dataset: PasDataset,
                     trials: int = 100,
                     alpha: float = 0.1,
                     summary: bool = True,
                     save_results: bool = False,
                     ci_methods: Optional[dict] = None,
                     ci_kwargs: Optional[dict] = None,
                     verbose: bool = True) -> pd.DataFrame:
    """Run benchmark experiments for confidence interval methods on any dataset.

    Args:
        dataset: Dataset to run experiments on
        trials: Number of trials to run
        alpha: Error level; CIs target 1-alpha coverage
        summary: Whether to print summary statistics
        save_results: Whether to save results to disk
        ci_methods: Dictionary of CI methods to use. If None, uses CORE_CI_METHODS
        ci_kwargs: Dictionary of keyword arguments to pass to each CI method.
            Each key is a CI method name, and the value is a dictionary of arguments.

    Returns:
        DataFrame containing coverage and width results for each CI method across trials
    """
    from intervals import CORE_CI_METHODS
    ci_methods = ci_methods or CORE_CI_METHODS
    col_names = []
    for name in ci_methods:
        col_names.extend([f"{name}_coverage", f"{name}_width"])

    ci_results = pd.DataFrame(columns=col_names)
    base_seed = dataset.split_seed

    for i in tqdm(range(trials), disable=not verbose):
        dataset.reload_data(split_seed=i + base_seed)
        true_theta = dataset.true_theta

        for ci_name, ci_func in ci_methods.items():
            kwargs = ci_kwargs.get(ci_name, {}) if ci_kwargs else {}
            ci = ci_func(dataset, alpha=alpha, **kwargs)  # shape (M, 2)
            coverage = get_coverage(true_theta, ci)
            width = get_avg_ci_width(ci)
            ci_results.loc[i, f"{ci_name}_coverage"] = coverage
            ci_results.loc[i, f"{ci_name}_width"] = width

    summary_text = [
        f"\nCI Results for {dataset.dataset_name} (alpha={alpha}):\n",
        f"Mean of metrics:\n {ci_results.mean().to_string(float_format='{:.8f}'.format)}",
        f"SD of metrics:\n {ci_results.std().to_string(float_format='{:.8f}'.format)}",
        f"SE of mean metrics:\n {ci_results.sem().to_string(float_format='{:.8f}'.format)}",
    ]

    if summary:
        print("\n".join(summary_text))

    if save_results:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        save_dir = BENCHMARK_FOLDER / f"{dataset.dataset_name}_ci_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        ci_results.to_csv(save_dir / "ci_results.csv")

        ci_args = {}
        for name, func in ci_methods.items():
            default_args = get_default_args(func)
            if ci_kwargs and name in ci_kwargs:
                default_args.update(ci_kwargs[name])
            ci_args[name] = default_args

        with open(save_dir / "ci_args.json", 'w') as f:
            json.dump(ci_args, f, indent=2)

        with open(save_dir / "summary.txt", 'w') as f:
            f.write("\n".join(summary_text))

    return ci_results


def visualize_benchmark(mse_results: pd.DataFrame,
                        dataset_name: str,
                        output_path: Optional[Path] = None):
    """ Visualize benchmark results for any dataset.

    Args:
        mse_results: DataFrame containing MSE results
        dataset_name: Name of dataset for plot title
        output_path: Path to save plot. If None, plot is displayed but not saved
    """
    import seaborn as sns
    
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
                                   output_path: Optional[Path] = None,
                                   verbose: bool = True):
    """ Run experiment mixing two estimators with different weights.

    Args:
        dataset: Dataset to run experiments on
        estimator1_name: Name of first estimator in CORE_ESTIMATORS
        estimator2_name: Name of second estimator in CORE_ESTIMATORS
        trials: Number of trials to run
        num_weights: Number of weight values to try between 0 and 1
        output_path: Path to save plot. If None, plot is displayed but not saved
        verbose: Whether to show a progress bar
    """
    assert estimator1_name in CORE_ESTIMATORS and estimator2_name in CORE_ESTIMATORS, \
        f"Estimators {estimator1_name} and {estimator2_name} must be in CORE_ESTIMATORS"

    weights = np.linspace(0, 1, num_weights)
    mse_results = np.zeros((trials, num_weights))

    for i in tqdm(range(trials), disable=not verbose):
        dataset.reload_data(split_seed=i + 12345)
        true_theta = dataset.true_theta

        estimates1 = CORE_ESTIMATORS[estimator1_name](dataset)
        estimates2 = CORE_ESTIMATORS[estimator2_name](dataset)

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
