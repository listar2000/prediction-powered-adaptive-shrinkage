from pas.experiments import run_benchmark
from pas.datasets.amazon_review import AmazonReviewDataset
from pas.config import DEFAULT_KWARGS

if __name__ == "__main__":
    dataset = AmazonReviewDataset(tuned=True)
    mse_results = run_benchmark(dataset, trials=200, summary=True, estimator_kwargs=DEFAULT_KWARGS)
