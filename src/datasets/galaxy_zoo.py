"""
Galaxy Zoo Dataset for morphology classification
- Data contains galaxy morphology labels and predictions
- Labels are binarized: 1 for spiral galaxies (original labels 3,4,5) and 0 otherwise
"""
from datasets.dataset import PasDataset
import numpy as np
import pandas as pd
from pathlib import Path
from config import DATA_PATHS
from typing import Tuple, List, Optional

# Get data paths from config
GALAXY_ZOO_FILE_PATH = DATA_PATHS["galaxy"]["predictions"]

SPIRAL_INDICES = set([3, 4, 5])


class GalaxyZooDataset(PasDataset):
    """ 
    Galaxy Zoo Dataset for morphology classification
    - Binary classification: spiral (1) vs non-spiral (0) galaxies
    - Problems are grouped by WVT_BIN regions
    """

    def __init__(self, file_path: Path = GALAXY_ZOO_FILE_PATH,
                 train_test_split: float = 0.2,
                 split_seed: int = 42,
                 verbose: bool = False):
        self.file_path = file_path
        self.train_test_split = train_test_split
        self.split_seed = split_seed
        super().__init__("Galaxy_zoo", verbose)

    def _read_raw_data(self) -> None:
        """ Read the raw data from CSV file and cache it.
        """
        assert self.file_path.exists(), f"File not found: {self.file_path}"
        assert self.file_path.suffix == ".csv", f"Invalid file format: {self.file_path.suffix}. Should be .csv"

        # Read CSV data
        df = pd.read_csv(self.file_path)

        # Convert labels to binary (1 for spiral galaxies, 0 otherwise)
        df['true_binary'] = df['true_label'].isin(SPIRAL_INDICES).astype(int)
        df['pred_binary'] = df['pred_label'].isin(SPIRAL_INDICES).astype(int)

        # Group by WVT_BIN and cache as list of arrays
        self.galaxy_groups = []
        for _, group in df.groupby('WVT_BIN'):
            # Store as 2xN array where row 0 is true labels and row 1 is predictions
            group_array = np.vstack([
                group['true_binary'].values,
                group['pred_binary'].values
            ])
            self.galaxy_groups.append(group_array)

    def load_data(self, train_test_split: float = None, split_seed: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
        """ Load the dataset with specified split parameters.
        """
        if not hasattr(self, "galaxy_groups"):
            self._read_raw_data()

        train_test_split = train_test_split or self.train_test_split
        split_seed = split_seed or self.split_seed

        pred_labelled, y_labelled, pred_unlabelled, y_unlabelled = [], [], [], []
        true_theta = []

        np.random.seed(split_seed)
        for galaxy_group in self.galaxy_groups:
            n = galaxy_group.shape[1]
            # shuffle and split the dataset
            indices = np.random.permutation(n)
            split_idx = int(n * train_test_split)

            # Split into labelled and unlabelled sets
            pred_labelled.append(galaxy_group[1, indices[:split_idx]])
            y_labelled.append(galaxy_group[0, indices[:split_idx]])
            pred_unlabelled.append(galaxy_group[1, indices[split_idx:]])
            y_unlabelled.append(galaxy_group[0, indices[split_idx:]])

            # Calculate true_theta as fraction of spiral galaxies in unlabelled set
            true_theta.append(np.mean(y_unlabelled[-1]))

        if self.verbose:
            print(
                f"`{self.dataset_name}` data loaded successfully from `{self.file_path}`")

        return pred_labelled, y_labelled, pred_unlabelled, y_unlabelled, np.array(true_theta)

    def reload_data(self, train_test_split: Optional[float] = None, split_seed: Optional[int] = None) -> None:
        """ Reload the dataset with new split parameters.
        """
        pred_labelled, y_labelled, pred_unlabelled, y_unlabelled, true_theta = self.load_data(
            train_test_split or self.train_test_split, split_seed or self.split_seed)
        self.set_metadata(pred_labelled, y_labelled,
                          pred_unlabelled, y_unlabelled, true_theta)
