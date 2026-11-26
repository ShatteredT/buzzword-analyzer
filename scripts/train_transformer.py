import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src.config import DATA_PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR, MODELS_DIR
from src.external import load_external
from src.metrics import compute_classification_metrics, save_metrics_json

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN, BATCH, EPOCHS, LR = 64, 32, 3, 2e-5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batch(tok, texts, device):
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN,
              padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def predict(model, tok, texts, device):
    model.eval()
    preds = []
    texts = list(texts)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
        for i in range(0, len(texts), 128):
            logits = model(**make_batch(tok, texts[i:i + 128], device)).logits
            preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds).numpy()


def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
    val = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")
    test = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")
    ext = load_external()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(device)

    counts = np.bincount(train["label"].values, minlength=2)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps_per_epoch = (len(train) + BATCH - 1) // BATCH
    total_steps = steps_per_epoch * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    texts = train["text"].tolist()
    labels = train["label"].values
    rng = np.random.RandomState(RANDOM_SEED)

    best_f1, best_state = -1.0, None
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train))
        total = 0.0
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            batch = make_batch(tok, [texts[j] for j in idx], device)
            y = torch.tensor(labels[idx]).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        m = compute_classification_metrics(val["label"], predict(model, tok, val["text"], device))
        print(f"epoch {epoch}: loss={total / len(train):.4f}, val macro_f1={m['macro_f1']:.4f}")
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_time = time.perf_counter() - t0

    model.load_state_dict(best_state)
    out_dir = MODELS_DIR / "stage3_distilbert"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    pred = predict(model, tok, test["text"], device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_time = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters())
    metrics = compute_classification_metrics(test["label"], pred)
    metrics.update({
        "stage": "3_distilbert_finetune",
        "model_name": MODEL_NAME,
        "val_macro_f1": best_f1,
        "train_time_sec": train_time,
        "inference_time_sec_per_1000": infer_time / len(test) * 1000,
        "n_params": n_params,
        "model_size_kb": n_params * 4 / 1024,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
    })
    save_metrics_json(metrics, REPORTS_DIR / "metrics_stage3_transformer.json")
    errors = test.assign(pred=pred)
    errors[errors["label"] != errors["pred"]].to_csv(REPORTS_DIR / "errors_stage3.csv", index=False)

    ext_pred = predict(model, tok, ext["text"], device)
    ext_metrics = compute_classification_metrics(ext["label"], ext_pred)
    save_metrics_json(ext_metrics, REPORTS_DIR / "metrics_stage3_external.json")
    ext.assign(pred=ext_pred).query("label != pred").to_csv(
        REPORTS_DIR / "errors_stage3_external.csv", index=False)
    print(f"EXTERNAL: accuracy={ext_metrics['accuracy']:.4f}, macro_f1={ext_metrics['macro_f1']:.4f}")

    print(f"\nTEST: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    print("confusion_matrix:", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()