from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_PATHS = {
    "amazon": {
        "tuned": PROJECT_ROOT / "data/amazon/amazon_review.h5",
        "raw": PROJECT_ROOT / "data/amazon/amazon_review_raw.h5"
    },
    "galaxy": {
        "predictions": PROJECT_ROOT / "data/galaxy/galaxy_test_prediction.csv"
    },
    "lmarena": {
        # Both files hold the same 40,663 pairwise comparisons and differ only
        # in the `prediction` column: `binary` is the judge's hard 0/1 call,
        # `bt_prob` the Bradley-Terry win probability. Everything downstream
        # (diagnostics, CI benchmarks, the docs notes) uses the Bradley-Terry
        # predictor, so `cleaned` resolves there rather than to the binary file.
        "binary": PROJECT_ROOT / "data/lmarena/clean_data/clean_summary.csv",
        "bt_prob": PROJECT_ROOT / "data/lmarena/clean_data/clean_summary_v2.csv",
        "cleaned": PROJECT_ROOT / "data/lmarena/clean_data/clean_summary_v2.csv",
    }
}

# Default kwargs for different experiments
DEFAULT_KWARGS = {
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
