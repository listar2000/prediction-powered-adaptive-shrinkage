# 📈 _PAS_: Prediction-Powered Adaptive Shrinkage Estimation

> Official repository for the ICML 2025 paper: [Prediction-Powered Adaptive Shrinkage Estimation](https://icml.cc/virtual/2025/poster/46514) by **Sida Li** & **Nikolaos Ignatiadis** (Data Science Institute, The University of Chicago). 

## TLDR: why might you want to use _PAS_?

- You just read the paper, and would love to see the implementation.
- You are from a **statistical background**, and would love to see how black-box ML predictions can make _compound mean estimation_ much more accurate and data-efficient.
- You are a **ML practitioner**, such as an NLP researcher, and once you have a good ML model at hand, you would love to see how to perform downstream statistical estimation with its predictions. A good example of this is the popular `LLM-as-ajudge` setting [We will release a notebook on this soon!].

## Setup and Installation
1. Please make sure you have `python >= 3.8` installed.
2. Clone this repository and install the dependencies:
```bash
git clone https://github.com/listar2000/prediction-powered-adaptive-shrinkage.git
cd prediction-powered-adaptive-shrinkage
pip install -r requirements.txt
```
3. [Optional] If you are only interested in the estimator implementation, you can simply go to the `src/estimators` folder and see all the implementation there (see directory structure below).

4. To test the installation, you can try to run the following demo script:
```bash
python src/scripts/run_galaxy_zoo.py
```

You should then see a progress bar as we simulate many repeated runs. After that, several metrics for the estimators will be printed out.

## Directory Structure
- `src/`: Contains all source code implementations
  - `estimators/`: Directory containing all estimation method implementations
    - `pas_estimators.py`: PAS estimators
    - `ppi_estimators.py`: PPI estimators
    - `simple_estimators.py`: Basic statistical estimators
    - `uni_pas_estimators.py`: Univariate PAS estimators
    - `legacy_estimators.py`: Legacy estimators (do not use)
  - `experiments.py`: Experiment configurations and setup
  - `utils.py`: Utility functions
  - `run_timing_benchmark.py`: Timing benchmark script
  - `datasets/`: Directory for dataset-specific code
    - `dataset.py`: Base class for dataset handling
    - `amazon_review.py`: Amazon Food Review dataset implementation
    - `galaxy_zoo.py`: Galaxy Zoo dataset implementation
    - `synthetic_model.py`: Synthetic Gaussian dataset implementation
  - `scripts/`: Directory for experiment scripts
    - `run_amazon_review.py`: Amazon Review dataset experiments
    - `run_galaxy_zoo.py`: Galaxy Zoo dataset experiments
    - `run_synthetic.py`: Synthetic dataset experiments
- `data/`: Directory for storing datasets
- `requirements.txt`: Python package dependencies

## A Unified Dataset Interface

_PAS_ is designed to be dataset-agnostic and work with any source dataset that contains **compound mean estimation problems**<sup>[1](#fn1)</sup>
. Therefore, we define a unified dataset interface called `PasDataset` in `src/datasets/dataset.py` -- every custom dataset should inherit from this class and implement the `load_data` method:
```python
def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
    """ Handle the logic for loading the dataset. This method should be implemented by the subclass.
    Returns:
      pred_labelled (List[np.ndarray]): \
          list of predictions for labelled data for each problem.

      y_labelled (List[np.ndarray]): \
          list of true responses for labelled data for each problem.

      pred_unlabelled (List[np.ndarray]): \ 
          list of predictions for unlabelled data for each problem.

      y_unlabelled (List[np.ndarray]): \
          list of true responses for unlabelled data for each problem.

      true_theta (np.ndarray): \
          the true theta for each problem.
    """
    pass
```

We have provided a few example dataset implementations (these are also examples mentioned in the paper):
- `src/datasets/synthetic_model.py`: Synthetic Gaussian dataset
- `src/datasets/amazon_review.py`: Amazon Review dataset
- `src/datasets/galaxy_zoo.py`: Galaxy Zoo dataset

One benefit of this unified interface is that once you provide the implementation of `load_data`, you do not need to worry about other data loading details, such as verification of data format and how we organize the data for estimators to consume.

<details>
<summary>
💡 Special note about the <code>reload_data</code> method
</summary>

The `reload_data` method is used to reload the dataset, which is a very common operation when we want to repeat the same experiment with different random seeds. There is a default implementation in the `PasDataset` class, which simply calls the `load_data` method with the same parameters. However, if you want to implement a custom dataset, you should override this method to reload the dataset with new split parameters.

</details>

## A Unified Estimator Interface

_PAS_ offers a very comprehensive suite of estimators for compound mean estimation problems. The directory structure above already gives an overview, but concretely, we have:

- `src/estimators/simple_estimators.py`: Basic statistical estimators like prediction mean and classical estimator.
- `src/estimators/ppi_estimators.py`: PPI and PPI++ estimators [[Angeloulous et al. 2024]](https://arxiv.org/abs/2311.01453)
- `src/estimators/pas_estimators.py`: _PAS_ estimators [[Ours]](https://arxiv.org/abs/2507.15500) & Shrinkage-only estimators [[Xie et al. 2012]](http://stat.wharton.upenn.edu/~lbrown/Papers/2012e%20SURE%20estimates%20for%20a%20heteroscedastic%20hierarchical%20model.pdf)
- `src/estimators/uni_pas_estimators.py`: Univariate _PAS_ estimators [[Ours (appendix)]](https://arxiv.org/abs/2507.15500)


In a nutshell, all estimators look like this:
```python
def estimator_name(dataset: PasDataset, **kwargs) -> np.ndarray:
    pass
```
which takes in a `PasDataset` object and returns a numpy array of estimates, whose length should equal the **number of problems** in the dataset. The `**kwargs` is used to pass in any additional arguments to the estimator. Sometimes more than one `np.ndarray` is returned, e.g. when we also want to return the shrinkage weights/levels for each problem.

## Best Practices for Running Experiments

#### Demo: comparing estimators for the `Galaxy Zoo` dataset

We have prepared an easy script to reproduce the paper experiments for the `Galaxy Zoo` dataset. You can run it by:
```bash
python src/scripts/run_galaxy_zoo.py
```

This also serves as a good example of how to compare estimators through running repeated experiments. First of all, we provide a handy function `run_benchmark` in `src/experiments.py` that can run repeated experiments on any dataset. The usage is as follows:
```python
from experiments import run_benchmark
from datasets.galaxy_zoo import GalaxyZooDataset
from estimators import CORE_ESTIMATORS
from config import DEFAULT_KWARGS

dataset = GalaxyZooDataset()

run_benchmark(dataset,
              trials=100,
              summary=True,
              save_results=False,
              estimators=CORE_ESTIMATORS,
              estimator_kwargs=DEFAULT_KWARGS)
```

Here, the arguments are:
- `dataset`: the dataset object
- `trials`: the number of repeated experiments to run
- `summary`: whether to print summary statistics
- `save_results`: whether to save the results
- `estimators`: a dictionary`<name, estimator_object>` of estimators to use. Since we have so many different estimators but you might only want to compare a few of them, you can pass in a subset of the estimators.
- `estimator_kwargs`: a dictionary `<name, kwargs>` of keyword arguments to pass to each estimator. Each name should match the name of the estimator in the `estimators` dictionary, and the value is the (optional) arguments to pass to that estimator (see the estimator signature above).

You can also use the `run_benchmark_timing` function to time the execution of each estimator.

## Roadmap:

- [x] Clean up and reorg the codebase, rewrite `README.md`.
- [ ] Add estimators from the [Regression for the mean [Erye & Madras 2025]](https://arxiv.org/abs/1207.0023) paper.
- [ ] Add `LLM-as-a-judge` dataset and notebooks.
- [ ] Refactor some estimator design to reuse shared code chunks.

## Citation

```bibtex
@inproceedings{LiIgnatiadis2025prediction,
  title = {Prediction-Powered Adaptive Shrinkage Estimation},
  author = {Sida Li and Nikolaos Ignatiadis},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)},
  year = {2025},
  note = {Poster presentation},
  url = {https://icml.cc/virtual/2025/poster/46514}
}
```

---
### Footnotes

<a name="fn1">1</a>: please refer to the [paper](https://arxiv.org/abs/2507.15500) for the definition of many concepts, such as "compound mean estimation problems", "power-tuning parameter", "shrinkage-to-mean", etc.