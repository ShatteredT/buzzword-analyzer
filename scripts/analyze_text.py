import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import MODELS_DIR
from train_nn import BiGRUClassifier, encode

MAX_LEN_TRANSFORMER = 64
NAMES = {
    "s1": "Стадія 1 (TF-IDF + LogReg)",
    "s2": "Стадія 2 (BiGRU)",
    "s3": "Стадія 3 (DistilBERT)",
}


class Analyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.s1 = joblib.load(MODELS_DIR / "stage1_tfidf_logreg.joblib")

        ckpt = torch.load(MODELS_DIR / "stage2_bigru.pt", map_location="cpu")
        self.vocab = ckpt["vocab"]
        self.s2 = BiGRUClassifier(len(self.vocab))
        self.s2.load_state_dict(ckpt["state_dict"])
        self.s2.to(self.device).eval()

        path = MODELS_DIR / "stage3_distilbert"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.s3 = AutoModelForSequenceClassification.from_pretrained(path)
        self.s3.to(self.device).eval()

    def p_s1(self, texts):
        return self.s1.predict_proba(texts)[:, 1]

    def p_s2(self, texts):
        ids, lengths = encode(texts, self.vocab)
        with torch.no_grad():
            logits = self.s2(ids.to(self.device), lengths)
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    def p_s3(self, texts):
        enc = self.tok(texts, truncation=True, max_length=MAX_LEN_TRANSFORMER,
                       padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.s3(**enc).logits
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    def predict(self, texts):
        return {"s1": self.p_s1(texts), "s2": self.p_s2(texts), "s3": self.p_s3(texts)}

    def explain_s1(self, text, k=5):
        vec = self.s1.named_steps["tfidfvectorizer"]
        clf = self.s1.named_steps["logisticregression"]
        x = vec.transform([text])
        contrib = x.multiply(clf.coef_[0]).tocoo()
        names = vec.get_feature_names_out()
        pairs = sorted(zip(contrib.col, contrib.data), key=lambda p: -abs(p[1]))[:k]
        return [(names[c], float(v)) for c, v in pairs]


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 3]


def label(p):
    return "корпоративна (1)" if p >= 0.5 else "проста (0)"


def analyze(an, text, whole):
    units = [text.strip()] if whole else split_sentences(text)
    if not units:
        print("Порожній текст.")
        return
    probs = an.predict(units)

    for i, u in enumerate(units):
        print(f"\n[{i + 1}] {u}")
        for key in ("s1", "s2", "s3"):
            p = float(probs[key][i])
            print(f"    {NAMES[key]:30s} P(corporate)={p:.2f} -> {label(p)}")
        expl = an.explain_s1(u)
        if expl:
            terms = ", ".join(f"{t} ({v:+.2f})" for t, v in expl)
            print(f"    Внесок ознак (Стадія 1; + до corporate, - до plain): {terms}")

    print("\nПідсумок за текстом (середнє P(corporate) по реченнях):")
    for key in ("s1", "s2", "s3"):
        mean = float(np.mean(probs[key]))
        print(f"  {NAMES[key]:30s} {mean:.2f} -> {label(mean)}")
    print("\nУвага: моделі навчено на синтетичному датасеті; на зовнішньому тесті "
          "macro-F1 становив 0,55–0,83. Результат орієнтовний.")


def main():
    parser = argparse.ArgumentParser(description="Аналіз тексту трьома моделями")
    parser.add_argument("--text", help="текст для аналізу")
    parser.add_argument("--file", help="шлях до .txt файлу (UTF-8)")
    parser.add_argument("--whole", action="store_true",
                        help="не розбивати на речення (довгий текст обрізається до 64 токенів)")
    args = parser.parse_args()

    an = Analyzer()
    if args.file:
        analyze(an, Path(args.file).read_text(encoding="utf-8"), args.whole)
    elif args.text:
        analyze(an, args.text, args.whole)
    else:
        print("Інтерактивний режим. Введіть текст і Enter; порожній рядок або 'q' — вихід.")
        while True:
            line = input("\n> ").strip()
            if not line or line.lower() == "q":
                break
            analyze(an, line, args.whole)


if __name__ == "__main__":
    main()