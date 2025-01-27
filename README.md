# Prediction-Powered Adaptive Shrinkage

This repository contains the implementation of prediction-powered adaptive shrinkage estimator.

## Directory Structure
- `src/`: Contains all source code implementations
  - `estimators.py`: Core implementation of estimation methods
  - `experiments.py`: Experiment configurations and setup
  - `utils.py`: Utility functions
  - `datasets/`: Directory for dataset-specific code
    - `dataset.py`: Base class for dataset handling
    - `amazon_review.py`: Amazon Food Review dataset implementation
    - `galaxy_zoo.py`: Galaxy Zoo dataset implementation
    - `synthetic_model.py`: Synthetic Gaussian dataset implementation
  - `scripts/`: Directory for scripts
    - `run_*.py`: Scripts for running different experiments
- `data/`: Directory for storing datasets (not included in repository)

## Setup and Installation
1. Install the required packages:
```bash
pip install -r src/requirements.txt
```

## Main Components
- Estimator implementations in `src/estimators.py`
- Experiment runners for different datasets:
  - Amazon Review dataset: `src/scripts/run_amazon_review.py`
  - Galaxy Zoo dataset: `src/scripts/run_galaxy_zoo.py`
  - Synthetic Gaussian examples: `src/scripts/run_synthetic.py`
