from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


SPLIT_TO_ID = {"train": 0, "val": 1, "validation": 1, "test": 2}


def build_default_transform(image_size: int = 64, train: bool = False):
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def _first_existing(candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def resolve_default_paths() -> dict[str, str]:
    image_candidates = [
        "data/img_align_celeba_reduced",
        "data/img_align_celeba",
    ]
    attr_candidates = [
        "data/list_attr_celeba_reduced.csv",
        "data/list_attr_celeba.csv",
    ]
    partition_candidates = [
        "data/list_eval_partition_reduced.csv",
        "data/list_eval_partition.csv",
    ]

    return {
        "image_dir": _first_existing(image_candidates) or image_candidates[0],
        "attr_file": _first_existing(attr_candidates) or attr_candidates[0],
        "partition_file": _first_existing(partition_candidates) or partition_candidates[0],
    }


class CelebADataset(Dataset):
    def __init__(
        self,
        img_dir: str | Path,
        attr_file: str | Path,
        transform=None,
        target_type: str = "classification",
        target_column: str = "Smiling",
        classification_label_scheme: str = "signed",
        regression_columns: Iterable[str] | None = None,
        weights: Sequence[float] | None = None,
        partition_file: str | Path | None = None,
        split: str | None = None,
        image_column: str | None = None,
    ):
        self.img_dir = Path(img_dir)
        self.transform = transform or build_default_transform()
        self.target_type = target_type
        self.target_column = target_column
        self.classification_label_scheme = classification_label_scheme

        metadata = pd.read_csv(attr_file)
        if metadata.empty:
            raise ValueError(f"No samples found in {attr_file}")

        if image_column is None:
            image_column = "image_id" if "image_id" in metadata.columns else metadata.columns[0]
        if image_column not in metadata.columns:
            raise ValueError(f"Image column '{image_column}' not found in {attr_file}")
        self.image_column = image_column

        if partition_file and Path(partition_file).exists():
            partitions = pd.read_csv(partition_file)
            if image_column not in partitions.columns:
                first_column = partitions.columns[0]
                partitions = partitions.rename(columns={first_column: image_column})
            if "partition" not in partitions.columns:
                raise ValueError(f"Column 'partition' not found in {partition_file}")
            metadata = metadata.merge(
                partitions[[image_column, "partition"]],
                on=image_column,
                how="inner",
            )

        if split is not None:
            split_key = split.lower()
            if split_key not in SPLIT_TO_ID:
                raise ValueError(f"Unsupported split '{split}'. Use train, val or test.")
            if "partition" not in metadata.columns:
                raise ValueError(
                    "A partition file is required when requesting a named split."
                )
            metadata = metadata[metadata["partition"] == SPLIT_TO_ID[split_key]]

        metadata = metadata.reset_index(drop=True).copy()
        self.attribute_columns = [
            column
            for column in metadata.columns
            if column not in {self.image_column, "partition", "target"}
        ]

        if target_type in {"classification", "smile"}:
            if target_column not in metadata.columns:
                raise ValueError(
                    f"Target column '{target_column}' not found in {attr_file}"
                )
            target_values = metadata[target_column].astype(float).to_numpy()
            positive_class = target_values > 0
            if classification_label_scheme == "signed":
                metadata["target"] = np.where(positive_class, 1.0, -1.0).astype(np.float32)
            elif classification_label_scheme == "binary":
                metadata["target"] = positive_class.astype(np.float32)
            else:
                raise ValueError(
                    "classification_label_scheme must be 'signed' or 'binary'"
                )
            self.task = "classification"
        elif target_type == "regression":
            columns = list(regression_columns) if regression_columns is not None else self.attribute_columns
            missing_columns = [column for column in columns if column not in metadata.columns]
            if missing_columns:
                raise ValueError(f"Regression columns not found: {missing_columns}")
            weight_array = np.ones(len(columns), dtype=np.float32)
            if weights is not None:
                if len(weights) != len(columns):
                    raise ValueError(
                        "weights must have the same length as regression_columns"
                    )
                weight_array = np.asarray(weights, dtype=np.float32)
            values = metadata[columns].astype(np.float32).to_numpy()
            metadata["target"] = values @ weight_array
            self.task = "regression"
            self.regression_columns = columns
        else:
            raise ValueError("target_type must be 'classification', 'smile' or 'regression'")

        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_path = self.img_dir / str(row[self.image_column])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        target = torch.tensor(float(row["target"]), dtype=torch.float32)
        return image, target


def _split_indices(
    n_samples: int,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    return train_idx, val_idx, test_idx


def create_dataloaders(
    img_dir: str | Path,
    attr_file: str | Path,
    target_type: str,
    batch_size: int,
    image_size: int = 64,
    partition_file: str | Path | None = None,
    target_column: str = "Smiling",
    classification_label_scheme: str = "signed",
    regression_columns: Iterable[str] | None = None,
    weights: Sequence[float] | None = None,
    num_workers: int = 0,
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    augment: bool = False,
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    train_transform = build_default_transform(image_size=image_size, train=augment)
    eval_transform = build_default_transform(image_size=image_size, train=False)

    base_kwargs = {
        "img_dir": img_dir,
        "attr_file": attr_file,
        "target_type": target_type,
        "target_column": target_column,
        "classification_label_scheme": classification_label_scheme,
        "regression_columns": regression_columns,
        "weights": weights,
        "partition_file": partition_file,
    }

    if partition_file and Path(partition_file).exists():
        train_dataset = CelebADataset(
            transform=train_transform,
            split="train",
            **base_kwargs,
        )
        val_dataset = CelebADataset(
            transform=eval_transform,
            split="val",
            **base_kwargs,
        )
        test_dataset = CelebADataset(
            transform=eval_transform,
            split="test",
            **base_kwargs,
        )
    else:
        full_train_dataset = CelebADataset(transform=train_transform, **base_kwargs)
        full_eval_dataset = CelebADataset(transform=eval_transform, **base_kwargs)
        train_idx, val_idx, test_idx = _split_indices(
            len(full_train_dataset),
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        train_dataset = Subset(full_train_dataset, train_idx.tolist())
        val_dataset = Subset(full_eval_dataset, val_idx.tolist())
        test_dataset = Subset(full_eval_dataset, test_idx.tolist())

    pin_memory = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    sizes = {split: len(loader.dataset) for split, loader in loaders.items()}
    return loaders, sizes
