import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import REPORTS_DIR

STAGES = [
    ("1. TF-IDF + LogReg", "metrics_stage1_baseline.json", "metrics_stage1_external.json", "errors_stage1_external.csv"),
    ("2. BiGRU", "metrics_stage2_nn.json", "metrics_stage2_external.json", "errors_stage2_external.csv"),
    ("3. DistilBERT", "metrics_stage3_transformer.json", "metrics_stage3_external.json", "errors_stage3_external.csv"),
]


def load(name):
    return json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))


rows = []
for title, test_f, ext_f, _ in STAGES:
    t, e = load(test_f), load(ext_f)
    rows.append({
        "Стадія": title,
        "test accuracy": round(t["accuracy"], 4),
        "test macro-F1": round(t["macro_f1"], 4),
        "external accuracy": round(e["accuracy"], 4),
        "external macro-F1": round(e["macro_f1"], 4),
        "external confusion [[TN,FP],[FN,TP]]": str(e["confusion_matrix"]),
        "час навчання, с": round(t["train_time_sec"], 1),
        "інференс, с/1000": round(t["inference_time_sec_per_1000"], 3),
        "розмір моделі, МБ": round(t["model_size_kb"] / 1024, 2),
    })

table = pd.DataFrame(rows)
md = table.to_markdown(index=False)
(REPORTS_DIR / "comparison.md").write_text(md, encoding="utf-8")
print(md)

print("\n=== Помилки на зовнішньому тесті ===")
for title, _, _, err_f in STAGES:
    err = pd.read_csv(REPORTS_DIR / err_f)
    print(f"\n{title}: помилок {len(err)}; "
          f"хибно 'corporate' (gold=0): {(err.label == 0).sum()}, "
          f"хибно 'plain' (gold=1): {(err.label == 1).sum()}")
    for _, r in err.head(8).iterrows():
        print(f"  gold={r.label} pred={r.pred} | {r.text}")