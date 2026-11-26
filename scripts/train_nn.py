import random
import re
import sys
import time
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.external import load_external
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import DATA_PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR, MODELS_DIR
from src.metrics import compute_classification_metrics, save_metrics_json

MAX_LEN, MIN_FREQ = 64, 2
EMB, HID = 64, 64
BATCH, EPOCHS, PATIENCE, LR = 64, 30, 6, 3e-4
TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9]")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts):
    counts = Counter(tok for t in texts for tok in tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, n in counts.most_common():
        if n >= MIN_FREQ:
            vocab[word] = len(vocab)
    return vocab


def encode(texts, vocab):
    ids = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    lengths = np.zeros(len(texts), dtype=np.int64)
    for i, t in enumerate(texts):
        toks = [vocab.get(w, 1) for w in tokenize(t)][:MAX_LEN] or [1]
        ids[i, :len(toks)] = toks
        lengths[i] = len(toks)
    return torch.tensor(ids), torch.tensor(lengths)


class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB, padding_idx=0)
        self.drop = nn.Dropout(0.3)
        self.gru = nn.GRU(EMB, HID, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * HID, 2)

    def forward(self, x, lengths):
        e = self.drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=x.size(1))
        out = out.masked_fill((x == 0).unsqueeze(-1), -1e9)
        return self.fc(self.drop(out.max(dim=1).values))


def predict(model, ids, lengths, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(ids), 256):
            logits = model(ids[i:i + 256].to(device), lengths[i:i + 256])
            preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds).numpy()


def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
    val = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")
    test = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")

    vocab = build_vocab(train["text"])
    print("розмір словника:", len(vocab))
    tr_x, tr_l = encode(train["text"], vocab)
    va_x, va_l = encode(val["text"], vocab)
    te_x, te_l = encode(test["text"], vocab)
    tr_y = torch.tensor(train["label"].values)

    counts = np.bincount(train["label"].values, minlength=2)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32).to(device)

    model = BiGRUClassifier(len(vocab)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(tr_x, tr_l, tr_y), batch_size=BATCH, shuffle=True)

    best_f1, best_state, bad = -1.0, None, 0
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for x, l, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device), l), y.to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        m = compute_classification_metrics(val["label"], predict(model, va_x, va_l, device))
        print(f"epoch {epoch}: loss={total / len(train):.4f}, val macro_f1={m['macro_f1']:.4f}")
        if m["macro_f1"] > best_f1:
            best_f1, bad = m["macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print("рання зупинка")
                break
    train_time = time.perf_counter() - t0

    model.load_state_dict(best_state)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "vocab": vocab}, MODELS_DIR / "stage2_bigru.pt")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    pred = predict(model, te_x, te_l, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_time = time.perf_counter() - t0
    ext = load_external()
    ext_x, ext_l = encode(ext["text"], vocab)
    ext_pred = predict(model, ext_x, ext_l, device)
    ext_metrics = compute_classification_metrics(ext["label"], ext_pred)
    save_metrics_json(ext_metrics, REPORTS_DIR / "metrics_stage2_external.json")
    ext.assign(pred=ext_pred).query("label != pred").to_csv(
        REPORTS_DIR / "errors_stage2_external.csv", index=False)
    print(f"EXTERNAL: accuracy={ext_metrics['accuracy']:.4f}, macro_f1={ext_metrics['macro_f1']:.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    metrics = compute_classification_metrics(test["label"], pred)
    metrics.update({
        "stage": "2_bigru_from_scratch",
        "val_macro_f1": best_f1,
        "train_time_sec": train_time,
        "inference_time_sec_per_1000": infer_time / len(test) * 1000,
        "n_params": n_params,
        "model_size_kb": n_params * 4 / 1024,
        "vocab_size": len(vocab),
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
    })
    save_metrics_json(metrics, REPORTS_DIR / "metrics_stage2_nn.json")
    errors = test.assign(pred=pred)
    errors[errors["label"] != errors["pred"]].to_csv(REPORTS_DIR / "errors_stage2.csv", index=False)

    print(f"\nTEST: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    print("confusion_matrix:", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()