from datasets.synthetic_model import GaussianSyntheticDataset
from experiments import run_ci_benchmark

if __name__ == "__main__":
    kwargs = {
        "pt_ci": {
            "share_var": False,
        },
    }
    # Run CI benchmark on synthetic data
    dataset = GaussianSyntheticDataset(
        good_f=True, M=100, has_true_vars=True, split_seed=4321)
    dataset.additional_y_variance = 0.05
    ci_results = run_ci_benchmark(
        dataset, trials=200, alpha=0.1, summary=True, ci_kwargs=kwargs)
