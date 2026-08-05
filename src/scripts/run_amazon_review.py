"""
Experiment with the Amazon review ratings dataset

Reproduces the "Amazon (tuned f)" column of Table 3; pass `tuned=False` for the
"Amazon (base f)" column. Use `src/scripts/post-fix/run_post_fix_tables.py` to
get all three dataset columns at once, plus the LaTeX rendering.
"""
from pas.experiments import run_benchmark
from pas.datasets.amazon_review import AmazonReviewDataset
from pas.estimators import PAPER_TABLE3_ESTIMATORS
from pas.config import DEFAULT_KWARGS

if __name__ == "__main__":
    dataset = AmazonReviewDataset(tuned=True)
    mse_results = run_benchmark(dataset, trials=200, summary=True,
                                estimators=PAPER_TABLE3_ESTIMATORS,
                                estimator_kwargs=DEFAULT_KWARGS)
