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
    }
}
