"""
Experiment with Galaxy Zoo dataset
"""
from pas.datasets.galaxy_zoo import GalaxyZooDataset
from pas.experiments import run_benchmark
from pas.config import DEFAULT_KWARGS

if __name__ == "__main__":
    dataset = GalaxyZooDataset()
    mse_results = run_benchmark(dataset, trials=200, summary=True, estimator_kwargs=DEFAULT_KWARGS)