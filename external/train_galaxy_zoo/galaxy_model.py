"""Dataset, transforms, model, and training loop for the Galaxy Zoo release."""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms
from tqdm import tqdm


class GalaxyZooDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, images_dir: Path, transform=None):
        self.labels = frame[["galaxyID", "label1"]].copy().reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        galaxy_id = str(self.labels.iloc[index]["galaxyID"])
        image = Image.open(self.images_dir / f"{galaxy_id}.jpg").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(self.labels.iloc[index]["label1"])
        return image, label, int(galaxy_id)


def train_validate_split(
    frame: pd.DataFrame,
    train_size: float = 0.8,
    seed: int = 1234,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int(train_size * len(shuffled))
    return shuffled.iloc[:split], shuffled.iloc[split:]


def create_data_transforms():
    train_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.99, 1.01)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    evaluation_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, evaluation_transform


def create_resnet50(num_classes: int):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def train_model(
    model,
    train_loader,
    valid_loader,
    *,
    num_epochs: int,
    criterion,
    optimizer,
    scheduler,
    device,
):
    best_weights = deepcopy(model.state_dict())
    best_valid_accuracy = 0.0
    history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}

    started = time.time()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for inputs, labels, _ in tqdm(train_loader, desc=f"epoch {epoch + 1} train"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += int((predictions == labels).sum())

        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        with torch.no_grad():
            for inputs, labels, _ in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predictions = outputs.argmax(dim=1)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                valid_correct += int((predictions == labels).sum())

        train_epoch_loss = train_loss / len(train_loader.dataset)
        train_epoch_accuracy = train_correct / len(train_loader.dataset)
        valid_epoch_loss = valid_loss / len(valid_loader.dataset)
        valid_epoch_accuracy = valid_correct / len(valid_loader.dataset)
        history["train_loss"].append(train_epoch_loss)
        history["train_acc"].append(train_epoch_accuracy)
        history["valid_loss"].append(valid_epoch_loss)
        history["valid_acc"].append(valid_epoch_accuracy)
        if valid_epoch_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_epoch_accuracy
            best_weights = deepcopy(model.state_dict())
        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"train loss={train_epoch_loss:.4f}, train acc={train_epoch_accuracy:.4f}, "
            f"valid loss={valid_epoch_loss:.4f}, valid acc={valid_epoch_accuracy:.4f}"
        )

    model.load_state_dict(best_weights)
    print(f"Training completed in {(time.time() - started) / 60:.1f} minutes")
    return model, history
