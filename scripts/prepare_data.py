import re
import sys
import random
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from datasets import load_dataset

from src.config import DATA_PROCESSED_DIR, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO

DATASET_ID = "phxdev/corporate-speak-dataset"
TEXT_RE = re.compile(
    r"###\s*Instruction:\s*(.*?)\s*###\s*Input:\s*(.*?)\s*###\s*Response:\s*(.*)",
    re.S,
)
VERB_RE = re.compile(r"\b(transform|convert|translate|rewrite|make|turn)\b")


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def key(s):
    return norm(s).lower()


def direction(instruction):
    ins = instruction.lower()
    if "plain english" in ins or "plain language" in ins:
        return "to_plain"
    if "corporate" in ins and VERB_RE.search(ins):
        return "to_corp"
    return None


def extract_fields(row):
    ins = norm(row.get("instruction"))
    inp = norm(row.get("input"))
    out = norm(row.get("output"))
    if not (ins and inp and out):
        m = TEXT_RE.search(row.get("text") or "")
        if m:
            ins, inp, out = (norm(g) for g in m.groups())
    return ins, inp, out


def instruction_family(ins):
    return re.sub(r"\(.*?\)", "(...)", ins.lower()).strip()


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def main():
    ds = load_dataset(DATASET_ID)
    pairs = []  # (plain, corp)
    families = Counter()
    n_rows = 0

    for split in ds.values():
        for row in split:
            n_rows += 1
            ins, inp, out = extract_fields(row)
            if not (ins and inp and out):
                families[("<empty>", "skip")] += 1
                continue
            d = direction(ins)
            families[(instruction_family(ins), d or "skip")] += 1
            if d == "to_plain":
                pairs.append((out, inp))
            elif d == "to_corp":
                pairs.append((inp, out))

    print(f"Рядків у датасеті: {n_rows}")
    print("\nІнструкції (топ-25) та їхня категорія:")
    for (fam, cat), n in families.most_common(25):
        print(f"  {n:6d}  {cat:9s}  {fam[:90]}")

    # Чищення пар
    seen, clean = set(), []
    for plain, corp in pairs:
        kp, kc = key(plain), key(corp)
        if kp == kc or len(kp) < 3 or len(kc) < 3:
            continue
        if (kp, kc) in seen:
            continue
        seen.add((kp, kc))
        clean.append((plain, corp))
    print(f"\nПар після чищення: {len(clean)}")

    # Мітки та конфлікти
    labels, original = {}, {}
    conflicts = set()
    for plain, corp in clean:
        for text, lab in ((plain, 0), (corp, 1)):
            k = key(text)
            original.setdefault(k, text)
            if k in labels and labels[k] != lab:
                conflicts.add(k)
            labels[k] = lab
    for k in conflicts:
        labels.pop(k, None)
    print(f"Унікальних текстів: {len(labels)} (конфліктних мітки видалено: {len(conflicts)})")

    # Групи пов'язаних текстів (union-find)
    parent = {k: k for k in labels}
    for plain, corp in clean:
        kp, kc = key(plain), key(corp)
        if kp in parent and kc in parent:
            parent[find(parent, kp)] = find(parent, kc)
    groups = {}
    for k in labels:
        groups.setdefault(find(parent, k), []).append(k)
    group_list = list(groups.values())
    largest = max(len(g) for g in group_list)
    print(f"Груп: {len(group_list)}, найбільша група: {largest} текстів")
    if largest > 0.05 * len(labels):
        print("УВАГА: одна група займає >5% даних, розбиття може бути нерівним.")

    # Розбиття за групами
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(group_list)
    total = len(labels)
    train_n, val_n = TRAIN_RATIO * total, VAL_RATIO * total
    splits = {"train": [], "val": [], "test": []}
    count = 0
    for g in group_list:
        if count < train_n:
            name = "train"
        elif count < train_n + val_n:
            name = "val"
        else:
            name = "test"
        splits[name].extend(g)
        count += len(g)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, keys in splits.items():
        df = pd.DataFrame(
            {"text": [original[k] for k in keys], "label": [labels[k] for k in keys]}
        ).sample(frac=1, random_state=RANDOM_SEED)
        df.to_csv(DATA_PROCESSED_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} рядків, клас 0: {(df.label == 0).sum()}, клас 1: {(df.label == 1).sum()}")


if __name__ == "__main__":
    main()