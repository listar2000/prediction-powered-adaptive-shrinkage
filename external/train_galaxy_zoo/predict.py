"""Run the trained Galaxy Zoo ResNet50 on the held-out WVT bins."""

from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from galaxy_model import GalaxyZooDataset, create_data_transforms, create_resnet50


HERE = Path(__file__).resolve().parent
ASSETS_DIR = HERE.parent / "assets" / "galaxy_zoo"


def asset_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ASSETS_DIR / path


def main() -> None:
    config = yaml.safe_load((HERE / "config.yaml").read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_frame = pd.read_csv(asset_path(config["test_csv"]))
    _, evaluation_transform = create_data_transforms()
    test_data = GalaxyZooDataset(
        test_frame, asset_path(config["images_dir"]), evaluation_transform
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = create_resnet50(config["num_classes"])
    model.load_state_dict(torch.load(asset_path(config["model_path"]), map_location=device))
    model.to(device).eval()
    galaxy_ids, true_labels, predicted_labels = [], [], []
    with torch.no_grad():
        for images, labels, ids in tqdm(test_loader, desc="predict"):
            predictions = model(images.to(device)).argmax(dim=1)
            galaxy_ids.extend(ids.tolist())
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.cpu().tolist())

    predictions = pd.DataFrame({
        "galaxyID": galaxy_ids,
        "true_label": true_labels,
        "pred_label": predicted_labels,
    })
    predictions = predictions.merge(
        test_frame[["galaxyID", "dr7objid", "WVT_BIN"]],
        on="galaxyID",
        how="left",
    )
    output = asset_path(config["prediction_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    accuracy = (predictions["true_label"] == predictions["pred_label"]).mean()
    print(f"Predictions saved to {output}")
    print(f"Held-out accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
