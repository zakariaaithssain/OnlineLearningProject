from __future__ import annotations

import argparse
import sys

from experiment_spec import EXPERIMENTAL_FRAME


STATUS_DONE = [
    "Deux architectures CNN existent deja dans le repo: `CNN1` et `CNN2`.",
    "Un notebook `tp1_related.ipynb` couvre surtout TP1: exploration du sous-jeu CelebA, creation d'un split et premiere analyse de complexite.",
    "Le fichier `utils.py` contient des briques pour TP1 a TP5: covering number, line search, validation croisee, regularisation, metriques, projection, normes et regret.",
    "Le repo dispose maintenant d'un cadre experimental fige pour le dataset, les deux CNN et les deux taches.",
    "La classification est alignee sur le PDF: score brut, labels signes {-1, +1} et perte hinge.",
    "Training is done for both tasks. "
]

STATUS_MISSING = [
    "La couverture TP2-TP5 existe surtout comme utilitaires de bas niveau, pas encore comme demonstration complete reliee aux CNN dans un rapport ou notebook final.",
]

ROADMAP = [
    "Comparer les deux architectures sur les deux taches avec les memes splits, seeds et hyperparametres de base.",
    "Produire les artefacts de rendu: courbes de loss, tableau final train/val/test, interpretation TP1-TP5 et conclusion.",
]


def print_status():
    print("Etat actuel du mini-projet CNN")
    print()
    print("Cadre experimental fige:")
    print(
        f"- Dataset: {EXPERIMENTAL_FRAME['dataset']['name']} "
        f"({EXPERIMENTAL_FRAME['dataset']['type']})"
    )
    print(
        f"- CNN1: {EXPERIMENTAL_FRAME['models']['CNN1']} | "
        f"CNN2: {EXPERIMENTAL_FRAME['models']['CNN2']}"
    )
    print(
        f"- Regression: {EXPERIMENTAL_FRAME['regression']['description']}"
    )
    print(
        f"- Classification: {EXPERIMENTAL_FRAME['classification']['description']} "
        f"avec labels {EXPERIMENTAL_FRAME['classification']['labels']} "
        f"et loss {EXPERIMENTAL_FRAME['classification']['loss']}"
    )
    print()
    print("Deja fait:")
    for item in STATUS_DONE:
        print(f"- {item}")
    print()
    print("A faire:")
    for item in STATUS_MISSING:
        print(f"- {item}")
    print()
    print("Roadmap concrete:")
    for index, item in enumerate(ROADMAP, start=1):
        print(f"{index}. {item}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(description="Point d'entree du mini-projet CNN")
        parser.add_argument(
            "command",
            choices=["status", "train-classification", "train-regression"],
            help="Sous-commande a executer",
        )
        parser.print_help()
        return

    command = argv[0]
    command_args = argv[1:]

    if command == "status":
        print_status()
        return

    if command == "train-classification":
        from train_classification import build_parser, run_training

        args = build_parser().parse_args(command_args)
        run_training(args)
        return

    if command == "train-regression":
        from train_regression import build_parser, run_training

        args = build_parser().parse_args(command_args)
        run_training(args)
        return

    raise ValueError(f"Unsupported command: {command}")


if __name__ == "__main__":
    main()
