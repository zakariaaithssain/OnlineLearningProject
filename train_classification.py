from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim

from data_loader import create_dataloaders, resolve_default_paths
from experiment_spec import EXPERIMENTAL_FRAME
from train_common import (
    BinaryHingeLoss,
    build_model,
    count_parameters,
    evaluate_classification,
    prepare_output_dirs,
    save_json,
    save_loss_curve,
    select_device,
    set_seed,
    train_one_epoch,
)


def add_arguments(parser: argparse.ArgumentParser):
    defaults = resolve_default_paths()
    classification_spec = EXPERIMENTAL_FRAME["classification"]
    parser.add_argument("--image-dir", default=defaults["image_dir"], help="Dossier des images CelebA")
    parser.add_argument("--attr-file", default=defaults["attr_file"], help="CSV des attributs")
    parser.add_argument(
        "--partition-file",
        default=defaults["partition_file"],
        help="CSV des partitions train/val/test. Si absent, un split aléatoire est utilisé.",
    )
    parser.add_argument(
        "--target-column",
        default=classification_spec["target_column"],
        help="Colonne cible pour la classification binaire",
    )
    parser.add_argument(
        "--model",
        choices=["simple", "improved"],
        default=EXPERIMENTAL_FRAME["models"]["CNN1"],
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
    parser.add_argument("--output-dir", default="outputs/classification")
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
    output_dir, figures_dir = prepare_output_dirs(args.output_dir, args.model)
    classification_spec = EXPERIMENTAL_FRAME["classification"]

    loaders, dataset_sizes = create_dataloaders(
        img_dir=args.image_dir,
        attr_file=args.attr_file,
        partition_file=args.partition_file if Path(args.partition_file).exists() else None,
        target_type="classification",
        target_column=args.target_column,
        classification_label_scheme=classification_spec["label_scheme"],
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        augment=args.augment,
    )

    model = build_model(args.model, task="classification").to(device)
    criterion = BinaryHingeLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_state_path = output_dir / "best_model.pt"
    best_val_f1 = float("-inf")
    best_epoch = 0

    print(
        f"Training classification on {device} with {args.model} "
        f"({count_parameters(model)} params)"
    )
    print(f"Dataset sizes: {dataset_sizes}")
    print(
        "Experimental frame: "
        f"target={args.target_column}, "
        f"labels={classification_spec['labels']}, "
        f"loss={classification_spec['loss']}, "
        f"decision_rule={classification_spec['decision_rule']}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        train_metrics = evaluate_classification(model, loaders["train"], criterion, device)
        val_metrics = evaluate_classification(model, loaders["val"], criterion, device)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(epoch_metrics)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_state_path)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

    model.load_state_dict(torch.load(best_state_path, map_location=device))
    test_metrics = evaluate_classification(model, loaders["test"], criterion, device)

    summary = {
        "task": "classification",
        "model": args.model,
        "device": str(device),
        "experimental_frame": classification_spec,
        "best_epoch": best_epoch,
        "dataset_sizes": dataset_sizes,
        "history": history,
        "test_metrics": test_metrics,
        "config": vars(args),
    }
    config = {
        "model": args.model,
        "task": "classification",
        "target_column": args.target_column,
        "label_scheme": classification_spec["label_scheme"],
        "labels": classification_spec["labels"],
        "decision_rule": classification_spec["decision_rule"],
        "loss": classification_spec["loss"],
        "optimizer": optimizer.__class__.__name__,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "seed": args.seed,
        "augment": args.augment,
        "device": str(device),
    }

    save_json(output_dir / "config.json", config)
    save_json(output_dir / "metrics.json", summary)
    save_loss_curve(
        history,
        figures_dir / "loss_curve.png",
        train_key="train_loss",
        val_key="val_loss",
        title=f"Classification loss ({args.model})",
    )

    print(
        "Test metrics: "
        f"loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['accuracy']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1']:.4f}"
    )
    print(f"Artifacts saved under {output_dir}")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Entraînement CNN pour la classification binaire CelebA (score brut + hinge)"
    )
    return add_arguments(parser)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
