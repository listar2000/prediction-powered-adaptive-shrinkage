from datasets.dataset import PasDataset
from pathlib import Path
from config import DATA_PATHS
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np

LMARENA_FILE_PATH = Path(DATA_PATHS["lmarena"]["cleaned"])


class LMArenaDataset(PasDataset):
    """
    LMArena Dataset for judging human preferences in pairwise LLM comparisons.
    Raw data is based on the lmarena-ai/arena-human-preference-140k dataset.
    """
    def __init__(
        self, 
        file_path: Path = LMARENA_FILE_PATH, 
        train_test_split: float = 0.2, 
        split_seed: int = 42,
        num_problems: int = -1,  # take all problems
        verbose: bool = False
    ):
        self.file_path = file_path
        self.train_test_split = train_test_split
        self.split_seed = split_seed
        self.num_problems = num_problems
        super().__init__("LMArena", verbose)

    def _read_raw_data(self) -> None:
        """ Read the raw data from the file.
        """
        assert self.file_path.exists(), f"File not found: {self.file_path}"
        assert self.file_path.suffix == ".csv", f"Invalid file format: {self.file_path.suffix}. Should be .csv"
        raw_df = pd.read_csv(self.file_path)

        if self.num_problems > 0:
            total_problems = int(raw_df["group_id"].nunique())
            max_group_id = min(self.num_problems, total_problems) - 1  # the group id is 0-indexed
            if self.verbose:
                print(f"Using the first {max_group_id + 1} (out of {total_problems}) problems")
            raw_df = raw_df[raw_df["group_id"] <= max_group_id]
        
        self.problem_groups = []
        for _, group in raw_df.groupby("group_id"):
            group_array = np.vstack([
                group["winner"].values,
                group["prediction"].values
            ])
            self.problem_groups.append(group_array)
        
    def load_data(
        self, 
        train_test_split: Optional[float] = None, 
        split_seed: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
        """ Load the dataset with specified split parameters.
        """
        if not hasattr(self, "problem_groups"):
            self._read_raw_data()

        train_test_split = train_test_split or self.train_test_split
        split_seed = split_seed or self.split_seed

        pred_labelled, y_labelled, pred_unlabelled, y_unlabelled = [], [], [], []
        true_theta = []

        np.random.seed(split_seed)
        for problem_group in self.problem_groups:
            n = problem_group.shape[1]
            # shuffle and split the dataset
            indices = np.random.permutation(n)
            split_idx = int(n * train_test_split)

            # Split into labelled and unlabelled sets
            pred_labelled.append(problem_group[1, indices[:split_idx]])
            y_labelled.append(problem_group[0, indices[:split_idx]])
            pred_unlabelled.append(problem_group[1, indices[split_idx:]])
            y_unlabelled.append(problem_group[0, indices[split_idx:]])

            # Calculate true_theta as fraction of winning predictions in combined (labelled + unlabelled) set
            true_theta.append(np.mean(np.concatenate([y_labelled[-1], y_unlabelled[-1]])))

        return pred_labelled, y_labelled, pred_unlabelled, y_unlabelled, np.array(true_theta)

    def reload_data(self, train_test_split: Optional[float] = None, split_seed: Optional[int] = None) -> None:
        """ Reload the dataset with new split parameters.
        """
        pred_labelled, y_labelled, pred_unlabelled, y_unlabelled, true_theta = self.load_data(
            train_test_split or self.train_test_split, split_seed or self.split_seed)

        self.set_metadata(pred_labelled, y_labelled, pred_unlabelled, y_unlabelled, true_theta)