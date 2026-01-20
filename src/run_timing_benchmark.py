from experiments import run_benchmark, run_benchmark_timing
from datasets.amazon_review import AmazonReviewDataset
from pathlib import Path

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
    },
    "uni_pas": {
        "get_lambda": False,
        "get_omega": False,
        "cutoff": 0.999
    },
    "uni_pt": {
        "get_lambda": False
    }
}

if __name__ == "__main__":
    dataset = AmazonReviewDataset(good_review=True)

    # mse_results = run_benchmark(
    #     dataset, trials=200, summary=True, estimator_kwargs=kwargs)
    # visualize_benchmark(
    #     mse_results, dataset_name="Amazon Review", output_path=output_dir / "benchmark_amazon_bad_with_k_fold.png")

    timing_results = run_benchmark_timing(
        dataset, summary=True, save_results=True, estimator_kwargs=kwargs)
