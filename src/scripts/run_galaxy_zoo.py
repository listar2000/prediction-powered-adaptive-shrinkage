"""
Experiment with Galaxy Zoo dataset

Reproduces the "Galaxy" column of Table 3. Use
`src/scripts/post-fix/run_post_fix_tables.py` to get all three dataset columns
at once, plus the LaTeX rendering.
"""
from pas.datasets.galaxy_zoo import GalaxyZooDataset
from pas.experiments import run_benchmark
from pas.estimators import PAPER_TABLE3_ESTIMATORS
from pas.config import DEFAULT_KWARGS

if __name__ == "__main__":
    dataset = GalaxyZooDataset()
    mse_results = run_benchmark(dataset, trials=200, summary=True,
                                estimators=PAPER_TABLE3_ESTIMATORS,
                                estimator_kwargs=DEFAULT_KWARGS)
