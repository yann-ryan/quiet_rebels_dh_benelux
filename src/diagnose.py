"""
Diagnostic script for suspiciously high evaluation metrics.

Run from src/:
    python ../diagnose.py --task crowd
    python ../diagnose.py --task human
    etc.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from train import stratified_split

# ── Same configs as evaluate_models.py ───────────────────────────────────────
TASK_CONFIGS = {
    "human": {
        "data":       "../data/annotated/massa_labels2.csv",
        "models_dir": "../models/human",
        "test_size":  0.4, "seed": 42,
        "stratified": False,
        "labels":     ["NONHUMAN", "HUMAN"],
        "preprocess": lambda df: df[df["label"] != 2].reset_index(drop=True),
    },
    "sentiment": {
        "data":       "../data/annotated/human_labelled_sentiment.csv",
        "models_dir": "../models/sentiment",
        "test_size":  0.4, "seed": 1234,
        "stratified": False,
        "labels":     ["NEGATIVE", "POSITIVE", "NEUTRAL"],
        "preprocess": None,
    },
    "type": {
        "data":       "../data/annotated/human_labelled_type.csv",
        "models_dir": "../models/type",
        "test_size":  0.4, "seed": 42,
        "stratified": True,
        "labels":     ["CITIZENS", "MILITARY", "LOWER", "PEOPLE", "COLONIAL", "FACELESS"],
        "preprocess": lambda df: (
            df[df["label"] != 2]
            .assign(label=lambda d: d["label"].map({0:0,1:1,3:2,4:3,5:4,6:5}))
            .reset_index(drop=True)
        ),
    },
    "situation": {
        "data":       "../data/annotated/human_labelled_situation.csv",
        "models_dir": "../models/situation",
        "test_size":  0.4, "seed": 42,
        "stratified": True,
        "labels":     ["POLITICS", "ECONOMY", "RELIGION", "SOCIAL", "MILITARY"],
        "preprocess": None,
    },
    "crowd": {
        "data":       "../data/annotated/crowds_df2.csv",
        "models_dir": "../models/crowd",
        "test_size":  0.4, "seed": 42,
        "stratified": False,
        "labels":     ["ABSTRACT", "CROWD"],
        "preprocess": None,
    },
}


def check_duplicates(df, train_idx, test_idx):
    print("\n── 1. DUPLICATE CHECK ───────────────────────────────────────")
    n_dup = df["text"].duplicated().sum()
    print(f"  Duplicate rows in full dataset : {n_dup}  ({n_dup/len(df)*100:.1f}%)")

    train_texts = set(df.iloc[train_idx]["text"])
    test_texts  = set(df.iloc[test_idx]["text"])
    overlap     = train_texts & test_texts
    print(f"  Unique texts in train          : {len(train_texts)}")
    print(f"  Unique texts in test           : {len(test_texts)}")
    print(f"  Texts in BOTH train and test   : {len(overlap)}  ← should be 0")
    if overlap:
        print("  *** LEAKAGE DETECTED — examples in the summary below ***")
        sample = df[df["text"].isin(list(overlap)[:3])][["text","label"]]
        print(sample.to_string())


def check_class_distribution(df, train_idx, test_idx, label_names):
    print("\n── 2. CLASS DISTRIBUTION ────────────────────────────────────")
    for split_name, idx in [("Train", train_idx), ("Test", test_idx)]:
        counts = df.iloc[idx]["label"].value_counts().sort_index()
        print(f"  {split_name} ({len(idx)} samples):")
        for lbl, cnt in counts.items():
            name = label_names[lbl] if lbl < len(label_names) else str(lbl)
            bar  = "█" * int(cnt / len(idx) * 40)
            print(f"    {name:15s} ({lbl})  {cnt:4d}  {cnt/len(idx)*100:5.1f}%  {bar}")


def check_model(model_path, test_dataset, label_names, task_average):
    print(f"\n── 3. PER-CLASS BREAKDOWN  [{os.path.basename(model_path)}] ─────────")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    mdl.eval()

    preds, true = [], []
    with torch.no_grad():
        for ex in test_dataset:
            inp = tok(ex["text"], truncation=True, padding="max_length", return_tensors="pt").to(device)
            preds.append(mdl(**inp).logits.argmax(-1).item())
            true.append(ex["label"])

    # Per-class report
    print(classification_report(true, preds, target_names=label_names, zero_division=0))

    # Prediction distribution — catch "always predicts one class"
    from collections import Counter
    pred_counts = Counter(preds)
    print("  Prediction distribution:")
    for lbl in sorted(pred_counts):
        name = label_names[lbl] if lbl < len(label_names) else str(lbl)
        pct  = pred_counts[lbl] / len(preds) * 100
        print(f"    {name:15s}: {pred_counts[lbl]:4d}  ({pct:.1f}%)")

    # Baseline: majority class
    majority = Counter(true).most_common(1)[0][0]
    dummy_preds = [majority] * len(true)
    print(f"\n  Majority-class baseline F1 ({label_names[majority] if majority < len(label_names) else majority}): "
          f"{f1_score(true, dummy_preds, average=task_average, zero_division=0):.3f}")

    # Confusion matrix
    cm = confusion_matrix(true, preds)
    print("\n  Confusion matrix (rows=true, cols=predicted):")
    header = "             " + "  ".join(f"{n[:6]:>6}" for n in label_names)
    print(header)
    for i, row in enumerate(cm):
        name = label_names[i] if i < len(label_names) else str(i)
        print(f"  {name[:12]:12s} " + "  ".join(f"{v:6d}" for v in row))


def main(task, model_name=None):
    cfg = TASK_CONFIGS[task]
    df  = pd.read_csv(cfg["data"])
    if cfg["preprocess"]:
        df = cfg["preprocess"](df)

    print(f"\n══════════════ DIAGNOSTICS: {task.upper()} ══════════════")
    print(f"  Dataset: {cfg['data']}  ({len(df)} rows after preprocessing)")

    # Recreate same split as training/evaluation
    if cfg["stratified"]:
        split      = stratified_split(df, test_size=cfg["test_size"], seed=cfg["seed"])
        test_ds    = split["test"]
        train_idx  = list(range(len(split["train"])))
        test_idx   = list(range(len(split["train"]), len(split["train"]) + len(split["test"])))
        # Approximate index reconstruction for duplicate check
        n_test     = len(split["test"])
        n_train    = len(split["train"])
        train_idx  = list(range(n_train))
        test_idx   = list(range(n_train, n_train + n_test))
        df_reindexed = pd.concat([
            split["train"].to_pandas(),
            split["test"].to_pandas(),
        ]).reset_index(drop=True)
        check_duplicates(df_reindexed, train_idx, test_idx)
        check_class_distribution(df_reindexed, train_idx, test_idx, cfg["labels"])
    else:
        from sklearn.model_selection import train_test_split as sk_split
        train_df, test_df = sk_split(df, test_size=cfg["test_size"], random_state=cfg["seed"])
        train_idx = list(train_df.index)
        test_idx  = list(test_df.index)
        test_ds   = Dataset.from_pandas(test_df.reset_index(drop=True))
        check_duplicates(df, train_idx, test_idx)
        check_class_distribution(df, train_idx, test_idx, cfg["labels"])

    # Find models to evaluate
    models_dir = cfg["models_dir"]
    if model_name:
        candidates = [os.path.join(models_dir, model_name)]
    else:
        candidates = sorted([
            os.path.join(models_dir, d)
            for d in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, d))
        ])

    for mp in candidates:
        check_model(mp, test_ds, cfg["labels"], cfg["average"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task",  required=True, choices=list(TASK_CONFIGS.keys()))
    p.add_argument("--model", default=None, help="Specific model subdirectory to check (optional)")
    args = p.parse_args()
    main(args.task, args.model)
