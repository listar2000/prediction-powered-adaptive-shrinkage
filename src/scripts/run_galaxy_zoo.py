"""
Experiment with Galaxy Zoo dataset
"""
from datasets.galaxy_zoo import GalaxyZooDataset
from experiments import run_benchmark
from config import DEFAULT_KWARGS

if __name__ == "__main__":
    dataset = GalaxyZooDataset()
    mse_results = run_benchmark(dataset, trials=200, summary=True, estimator_kwargs=DEFAULT_KWARGS)