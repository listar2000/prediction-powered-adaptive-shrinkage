"""Train the eight-class Galaxy Zoo ResNet50 used by the PAS experiments."""

from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from galaxy_model import (
    GalaxyZooDataset,
    create_data_transforms,
    create_resnet50,
    train_model,
    train_validate_split,
)


HERE = Path(__file__).resolve().parent
ASSETS_DIR = HERE.parent / "assets" / "galaxy_zoo"


def asset_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ASSETS_DIR / path


def main() -> None:
    config = yaml.safe_load((HERE / "config.yaml").read_text())
    torch.manual_seed(config["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_frame, valid_frame = train_validate_split(
        pd.read_csv(asset_path(config["train_csv"])), seed=config["seed"]
    )
    train_transform, evaluation_transform = create_data_transforms()
    train_data = GalaxyZooDataset(
        train_frame, asset_path(config["images_dir"]), train_transform
    )
    valid_data = GalaxyZooDataset(
        valid_frame, asset_path(config["images_dir"]), evaluation_transform
    )
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_data, shuffle=False, **loader_kwargs)
    print(f"Training samples: {len(train_data)}; validation samples: {len(valid_data)}")

    model = create_resnet50(config["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=config["step_size"], gamma=config["gamma"]
    )
    model, history = train_model(
        model,
        train_loader,
        valid_loader,
        num_epochs=config["num_epochs"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    model_path = asset_path(config["model_path"])
    history_path = asset_path(config["history_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Model saved to {model_path}")
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
