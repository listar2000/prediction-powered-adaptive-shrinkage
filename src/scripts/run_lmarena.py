"""
Experiment with LMArena dataset
"""
from pas.datasets.lmarena import LMArenaDataset
from pas.experiments import run_benchmark
from pas.estimators import ALL_ESTIMATORS
from pas.config import DEFAULT_KWARGS

if __name__ == "__main__":
    dataset = LMArenaDataset(train_test_split=0.4)
    # make a copy of the default kwargs and set share_var to True for all estimators
    custom_kwargs = {}
    for k, v in DEFAULT_KWARGS.items():
        custom_kwargs[k] = v.copy()
        if "share_var" in custom_kwargs[k]:
            custom_kwargs[k]["share_var"] = False
    print(custom_kwargs)

    mse_results = run_benchmark(dataset, trials=200, summary=True, estimators=ALL_ESTIMATORS, estimator_kwargs=custom_kwargs)