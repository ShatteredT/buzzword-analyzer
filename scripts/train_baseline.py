import argparse
import io
import re
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.external import load_external
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.config import DATA_PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR, MODELS_DIR
from src.metrics import compute_classification_metrics, save_metrics_json


def normalize(text):
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize", action="store_true",
                        help="нижній регістр і видалення пунктуації (перевірка стильових артефактів)")
    args = parser.parse_args()
    suffix = "_normalized" if args.normalize else ""

    train = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
    val = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")
    test = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")
    if args.normalize:
        for df in (train, val, test):
            df["text"] = df["text"].map(normalize)

    best = None
    for c in (0.1, 1, 10, 100):
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(C=c, class_weight="balanced", max_iter=1000,
                               random_state=RANDOM_SEED),
        )
        pipe.fit(train["text"], train["label"])
        m = compute_classification_metrics(val["label"], pipe.predict(val["text"]))
        print(f"C={c}: val macro_f1={m['macro_f1']:.4f}")
        if best is None or m["macro_f1"] > best[0]:
            best = (m["macro_f1"], c)

    best_c = best[1]
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000,
                           random_state=RANDOM_SEED),
    )
    t0 = time.perf_counter()
    pipe.fit(train["text"], train["label"])
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = pipe.predict(test["text"])
    infer_time = time.perf_counter() - t0

    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / f"stage1_tfidf_logreg{suffix}.joblib")

    metrics = compute_classification_metrics(test["label"], pred)
    metrics.update({
        "stage": "1_baseline_tfidf_logreg" + suffix,
        "best_C": best_c,
        "val_macro_f1": best[0],
        "train_time_sec": train_time,
        "inference_time_sec_per_1000": infer_time / len(test) * 1000,
        "model_size_kb": len(buf.getvalue()) / 1024,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
    })
    save_metrics_json(metrics, REPORTS_DIR / f"metrics_stage1_baseline{suffix}.json")

    errors = test.assign(pred=pred)
    errors = errors[errors["label"] != errors["pred"]]
    errors.to_csv(REPORTS_DIR / f"errors_stage1{suffix}.csv", index=False)
    ext = load_external()
    ext_pred = pipe.predict(ext["text"])
    ext_metrics = compute_classification_metrics(ext["label"], ext_pred)
    save_metrics_json(ext_metrics, REPORTS_DIR / f"metrics_stage1_external{suffix}.json")
    ext.assign(pred=ext_pred).query("label != pred").to_csv(
        REPORTS_DIR / f"errors_stage1_external{suffix}.csv", index=False)
    print(f"EXTERNAL: accuracy={ext_metrics['accuracy']:.4f}, macro_f1={ext_metrics['macro_f1']:.4f}")
    print(f"\nTEST: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    print("confusion_matrix:", metrics["confusion_matrix"])
    print(f"Помилок: {len(errors)}; збережено в reports")


if __name__ == "__main__":
    main()