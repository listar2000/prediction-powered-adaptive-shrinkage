from pas.datasets.synthetic_model import GaussianSyntheticDataset
from pas.experiments import run_benchmark

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
    # Run basic benchmark experiments with Gaussian V2 Dataset.
    # m = 200 problems, n_j = 20, N_j = 80: the Table 2 / Figure 3 configuration.
    # Flip `good_f` to False for the flawed predictor f_2(x) = |x| column.
    dataset = GaussianSyntheticDataset(
        good_f=True, M=200, has_true_vars=True, split_seed=4321)
    mse_results = run_benchmark(
        dataset, trials=200, summary=True, estimator_kwargs=kwargs)
