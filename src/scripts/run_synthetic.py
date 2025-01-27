from datasets.gaussian_toy import GaussianV2Dataset
from experiments import run_benchmark

if __name__ == "__main__":
    kwargs = {
        "power_tuned_ppi": {
            "share_var": False,
            "get_lambdas": False
        },
        "eb_sure": {
            "get_lambdas": False,
            "share_var": False,
            "cutoff": 0.999
        },
        "sure_ppi": {
            "get_lambdas": False,
            "share_var": False,
            "cutoff": 0.999,
            "old_ver": False
        },
        "split_sure_ppi": {
            "split_ratio": 0.5,
            "cutoff": 0.999,
            "share_var": False,
            "get_lambdas": False
        }
    }
    # Run basic benchmark experiments with Gaussian V2 Dataset
    dataset = GaussianV2Dataset(good_f=True, has_true_vars=True, split_seed=4321)
    dataset.additional_y_variance = 0.05
    mse_results = run_benchmark(dataset, trials=200, summary=True, estimator_kwargs=kwargs)