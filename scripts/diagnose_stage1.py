import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import DATA_PROCESSED_DIR, RANDOM_SEED

train = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X = vec.fit_transform(train["text"])
clf = LogisticRegression(C=10, class_weight="balanced", max_iter=1000,
                         random_state=RANDOM_SEED).fit(X, train["label"])

names = np.array(vec.get_feature_names_out())
order = np.argsort(clf.coef_[0])
print("Топ-15 ознак класу 0 (plain):", list(names[order[:15]]))
print("Топ-15 ознак класу 1 (corporate):", list(names[order[-15:]]))

train["n_words"] = train["text"].str.split().str.len()
print("\nСередня довжина (слів) за класом:")
print(train.groupby("label")["n_words"].describe()[["mean", "50%", "max"]])

manual = [
    ("The server crashed because the disk was full.", 0),
    ("We missed the deadline because nobody owned the task.", 0),
    ("Please send me the report by Friday afternoon.", 0),
    ("I think this plan is too expensive for what we get.", 0),
    ("The meeting is at 3 pm in the main conference room.", 0),
    ("Sales dropped 12 percent last quarter due to the pricing change.", 0),
    ("Let's circle back offline to align on key deliverables going forward.", 1),
    ("We need to leverage our core competencies to unlock synergies.", 1),
    ("This initiative will move the needle on stakeholder engagement.", 1),
    ("Let's take a deep dive into the low-hanging fruit at our next touchpoint.", 1),
    ("We are pivoting to a holistic, best-in-class value proposition.", 1),
    ("Going forward, we will optimize bandwidth to drive actionable outcomes.", 1),
]
texts, gold = zip(*manual)
proba = clf.predict_proba(vec.transform(texts))[:, 1]
print("\nРучні приклади (gold / P(corporate) / текст):")
for g, p, t in zip(gold, proba, texts):
    flag = "" if (p > 0.5) == bool(g) else "  <-- ПОМИЛКА"
    print(f"  {g}  {p:.2f}  {t}{flag}")