import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_classification_metrics(y_true, y_pred):
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1_class_0": float(f1_per_class[0]),
        "f1_class_1": float(f1_per_class[1]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def save_metrics_json(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")