from datasets.dataset import PasDataset
import numpy as np
from pathlib import Path
import h5py
from config import DATA_PATHS

# Get data paths from config
AMAZON_FOOD_REVIEW_TUNED = Path(DATA_PATHS["amazon"]["tuned"])
AMAZON_FOOD_REVIEW_RAW = Path(DATA_PATHS["amazon"]["raw"])

class AmazonReviewDataset(PasDataset):
    """ 
    Amazon Food Review Dataset
    - raw data: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
    - pretrained model: https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
    """
    def __init__(self, file_path: Path = None, 
                 tuned: bool = True,
                 train_test_split: float = 0.2, 
                 split_seed: int = 42, 
                 verbose: bool = False):
        # override the `tuned` flag if the file path is provided
        if file_path is not None:    
            self.file_path = file_path
            dataset_name = "Amazon_food_custom"
        else:
            self.file_path = AMAZON_FOOD_REVIEW_TUNED if tuned else AMAZON_FOOD_REVIEW_RAW
            dataset_name = "Amazon_food_" + ("tuned" if tuned else "raw")
        self.train_test_split = train_test_split
        self.split_seed = split_seed
        super().__init__(dataset_name, verbose)

    def _read_raw_data(self) -> None:
        """ Read the raw data from the file.
        """
        assert self.file_path.exists(), f"File not found: {self.file_path}"
        assert self.file_path.suffix == ".h5", f"Invalid file format: {self.file_path.suffix}. Should be .h5"

        product_reviews = []
        with h5py.File(self.file_path, "r") as f:
            for product_id in f.keys():
                product_reviews.append(f[product_id][:])
        # cache the product reviews
        self.product_reviews = product_reviews

    def load_data(self, train_test_split: float = None, split_seed: int = None) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
        if not hasattr(self, "product_reviews"):
            self._read_raw_data()

        train_test_split = train_test_split or self.train_test_split
        split_seed = split_seed or self.split_seed

        pred_labelled, y_labelled, pred_unlabelled, y_unlabelled = [], [], [], []
        true_theta = []

        np.random.seed(split_seed)
        for product_reviews in self.product_reviews:
            n = product_reviews.shape[1]
            # shuffle and split the dataset
            indices = np.random.permutation(n)
            split_idx = int(n * train_test_split)
            pred_labelled.append(product_reviews[1, indices[:split_idx]])
            y_labelled.append(product_reviews[0, indices[:split_idx]])
            pred_unlabelled.append(product_reviews[1, indices[split_idx:]])
            y_unlabelled.append(product_reviews[0, indices[split_idx:]])

            # calculate the `true` theta through y_unlabelled
            true_theta.append(np.mean(y_unlabelled[-1]))
        
        if self.verbose:
            print(f"`{self.dataset_name}` data loaded successfully from `{self.file_path}`")

        return pred_labelled, y_labelled, pred_unlabelled, y_unlabelled, np.array(true_theta)


    def reload_data(self, train_test_split: float = 0.2, split_seed: int = 42) -> None:
        """ Reload the dataset with new split parameters.
        """        
        pred_labelled, y_labelled, pred_unlabelled, y_unlabelled, true_theta = self.load_data(train_test_split, split_seed)

        self.validate_data(pred_labelled, y_labelled, pred_unlabelled, y_unlabelled)
        self.set_metadata(pred_labelled, y_labelled, pred_unlabelled, y_unlabelled, true_theta)
