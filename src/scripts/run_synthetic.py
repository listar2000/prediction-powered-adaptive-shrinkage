from datasets.synthetic_model import GaussianSyntheticDataset
from experiments import run_benchmark

if __name__ == "__main__":
    kwargs = {
        "pt": {
            "share_var": False,
            "get_lambdas": False
        },
        "shrinkage_only": {
            "get_lambdas": False,
            "share_var": False,
            "cutoff": 0.999
        },
        "pas": {
            "get_lambdas": False,
            "share_var": False,
            "cutoff": 0.999,
        },
        "shrinkage_mean": {
            "cutoff": 0.999,
            "share_var": False,
            "get_lambdas": False
        }
    }
    # Run basic benchmark experiments with Gaussian V2 Dataset
    dataset = GaussianSyntheticDataset(
        good_f=False, has_true_vars=True, split_seed=4321)
    dataset.additional_y_variance = 0.05
    mse_results = run_benchmark(
        dataset, trials=200, summary=True, estimator_kwargs=kwargs)
