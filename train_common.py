from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from models.cnn_improved import CNN2
from models.cnn_simple import CNN1
from utils import accuracy, precision_recall_f1


MODEL_REGISTRY = {
    "simple": CNN1,
    "improved": CNN2,
}


class BinaryHingeLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, outputs, targets):
        losses = torch.clamp(self.margin - targets * outputs, min=0.0)
        return losses.mean()


def parse_float_list(raw_value: str | None):
    if raw_value is None or raw_value.strip() == "":
        return None
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_name_list(raw_value: str | None):
    if raw_value is None or raw_value.strip() == "":
        return None
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(model_name: str, task: str) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")
    output_type = "classification" if task == "classification" else "regression"
    return MODEL_REGISTRY[model_name](output_type=output_type)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: str | Path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_output_dirs(base_output_dir: str | Path, model_name: str):
    run_dir = ensure_dir(Path(base_output_dir) / model_name)
    figures_dir = ensure_dir(run_dir / "figures")
    return run_dir, figures_dir


def save_loss_curve(history, path: str | Path, train_key: str, val_key: str, title: str):
    epochs = [item["epoch"] for item in history]
    train_values = [item[train_key] for item in history]
    val_values = [item[val_key] for item in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_values, marker="o", label=train_key)
    plt.plot(epochs, val_values, marker="o", label=val_key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1)

        optimizer.zero_grad()
        outputs = model(inputs).view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_classification(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1)
        outputs = model(inputs).view(-1)
        loss = criterion(outputs, targets)

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = torch.where(outputs >= 0.0, 1, -1).cpu().numpy()
        y_pred.extend(predictions.tolist())
        y_true.extend(targets.cpu().numpy().astype(int).tolist())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred, pos_label=1)
    return {
        "loss": running_loss / max(total_samples, 1),
        "accuracy": float(accuracy(y_true, y_pred)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.no_grad()
def evaluate_regression(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    targets_list = []
    predictions_list = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1)
        outputs = model(inputs).view(-1)
        loss = criterion(outputs, targets)

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        targets_list.extend(targets.cpu().numpy().tolist())
        predictions_list.extend(outputs.cpu().numpy().tolist())

    y_true = np.asarray(targets_list, dtype=np.float32)
    y_pred = np.asarray(predictions_list, dtype=np.float32)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    variance = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / variance) if variance > 0 else 0.0

    return {
        "loss": running_loss / max(total_samples, 1),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }
