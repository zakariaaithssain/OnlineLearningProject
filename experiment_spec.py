from __future__ import annotations


EXPERIMENTAL_FRAME = {
    "dataset": {
        "name": "CelebA reduced",
        "type": "image attributes",
        "note": "Use the reduced CelebA split when available, otherwise fall back to CelebA.",
    },
    "models": {
        "CNN1": "simple",
        "CNN2": "improved",
    },
    "regression": {
        "name": "attribute_sum_regression",
        "description": "Predict the sum of all available CelebA attributes.",
        "target_column_mode": "all_attributes_sum",
        "loss": "mse",
        "selection_metric": "rmse",
    },
    "classification": {
        "name": "smiling_binary_classification",
        "description": "Predict whether the face is smiling from a raw scalar score.",
        "target_column": "Smiling",
        "label_scheme": "signed",
        "labels": [-1, 1],
        "decision_rule": "sign(score)",
        "loss": "hinge",
        "selection_metric": "f1",
    },
}
