from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_PATHS = {
    "amazon": {
        "tuned": PROJECT_ROOT / "data/amazon/amazon_review.h5",
        "raw": PROJECT_ROOT / "data/amazon/amazon_review_raw.h5"
    },
    "galaxy": {
        "predictions": PROJECT_ROOT / "data/galaxy/galaxy_test_prediction.csv"
    }
}

# Default kwargs for different experiments
DEFAULT_KWARGS = {
    "power_tuned_ppi": {
        "share_var": False,
        "get_lambdas": False
    },
    "eb_sure": {
        "get_lambdas": False,
        "share_var": False,
        "cutoff": 0.999
    },
    "sure_ppi": {
        "get_lambdas": False,
        "share_var": False,
        "cutoff": 0.999,
        "old_ver": False
    },
    "split_sure_ppi": {
        "split_ratio": 0.5,
        "cutoff": 0.999,
        "share_var": False,
        "get_lambdas": False
    }
}
