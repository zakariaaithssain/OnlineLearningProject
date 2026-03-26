from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim

from data_loader import create_dataloaders, resolve_default_paths
from experiment_spec import EXPERIMENTAL_FRAME
from train_common import (
    build_model,
    count_parameters,
    ensure_dir,
    evaluate_regression,
    parse_float_list,
    parse_name_list,
    save_json,
    select_device,
    set_seed,
    train_one_epoch,
)


def add_arguments(parser: argparse.ArgumentParser):
    defaults = resolve_default_paths()
    regression_spec = EXPERIMENTAL_FRAME["regression"]
    parser.add_argument("--image-dir", default=defaults["image_dir"], help="Dossier des images CelebA")
    parser.add_argument("--attr-file", default=defaults["attr_file"], help="CSV des attributs")
    parser.add_argument(
        "--partition-file",
        default=defaults["partition_file"],
        help="CSV des partitions train/val/test. Si absent, un split aléatoire est utilisé.",
    )
    parser.add_argument(
        "--regression-columns",
        default=None,
        help="Colonnes utilisées pour la cible de régression, séparées par des virgules. Par défaut: tous les attributs.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Poids de la cible de régression, séparés par des virgules. Même longueur que --regression-columns si fourni.",
    )
    parser.add_argument(
        "--model",
        choices=["simple", "improved"],
        default=EXPERIMENTAL_FRAME["models"]["CNN2"],
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda ou mps")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--augment", action="store_true", help="Active un flip horizontal aléatoire sur le train")
    parser.add_argument("--output-dir", default="outputs/regression")
    return parser


def _validate_input_paths(args):
    missing = [
        path
        for path in [args.image_dir, args.attr_file]
        if not Path(path).exists()
    ]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing dataset path(s): {joined}. "
            "Pass --image-dir/--attr-file explicitly or add the CelebA files under data/."
        )


def run_training(args):
    _validate_input_paths(args)
    set_seed(args.seed)
    device = select_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    regression_spec = EXPERIMENTAL_FRAME["regression"]

    regression_columns = parse_name_list(args.regression_columns)
    weights = parse_float_list(args.weights)

    loaders, dataset_sizes = create_dataloaders(
        img_dir=args.image_dir,
        attr_file=args.attr_file,
        partition_file=args.partition_file if Path(args.partition_file).exists() else None,
        target_type="regression",
        regression_columns=regression_columns,
        weights=weights,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        augment=args.augment,
    )

    model = build_model(args.model, task="regression").to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_state_path = output_dir / "best_model.pt"
    best_val_rmse = float("inf")
    best_epoch = 0

    print(
        f"Training regression on {device} with {args.model} "
        f"({count_parameters(model)} params)"
    )
    print(f"Dataset sizes: {dataset_sizes}")
    print(
        "Experimental frame: "
        f"target_mode={regression_spec['target_column_mode']}, "
        f"loss={regression_spec['loss']}, "
        f"selection_metric={regression_spec['selection_metric']}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        train_metrics = evaluate_regression(model, loaders["train"], criterion, device)
        val_metrics = evaluate_regression(model, loaders["val"], criterion, device)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
        }
        history.append(epoch_metrics)

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_state_path)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} "
            f"val_r2={val_metrics['r2']:.4f}"
        )

    model.load_state_dict(torch.load(best_state_path, map_location=device))
    test_metrics = evaluate_regression(model, loaders["test"], criterion, device)

    summary = {
        "task": "regression",
        "model": args.model,
        "device": str(device),
        "experimental_frame": regression_spec,
        "best_epoch": best_epoch,
        "dataset_sizes": dataset_sizes,
        "history": history,
        "test_metrics": test_metrics,
        "config": vars(args),
    }
    save_json(output_dir / "metrics.json", summary)

    print(
        "Test metrics: "
        f"loss={test_metrics['loss']:.4f} "
        f"mae={test_metrics['mae']:.4f} "
        f"rmse={test_metrics['rmse']:.4f} "
        f"r2={test_metrics['r2']:.4f}"
    )
    print(f"Artifacts saved under {output_dir}")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Entraînement CNN pour la régression CelebA")
    return add_arguments(parser)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
