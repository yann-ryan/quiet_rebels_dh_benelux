"""
Evaluate all fine-tuned models in a given directory against a labelled test set.

Usage:
    python evaluate_models.py --task human
    python evaluate_models.py --task crowd
    python evaluate_models.py --task sentiment
    python evaluate_models.py --task type
    python evaluate_models.py --task situation

Outputs (all written relative to the repo root):
    images/metrics_{task}.png
    images/confusion_matrices_{task}.png
    images/metrics_{task}.csv          # tidy long-format, ready for R/ggplot2
    images/confusion_{task}.csv        # tidy long-format confusion data for R
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train import stratified_split

# Output directory for all images and CSVs — relative to src/, so ../images/
IMAGES_DIR = "../images"


TASK_CONFIGS = {
    "human": {
        "data":       "../data/annotated/massa_labels2.csv",
        "models_dir": "../models/human",
        "test_size":  0.4,
        "seed":       42,
        "average":    "macro",
        "stratified": False,
        "labels":     ["NONHUMAN", "HUMAN"],
        "preprocess": lambda df: df[df["label"] != 2].reset_index(drop=True),
    },
    "sentiment": {
        "data":       "../data/annotated/human_labelled_sentiment.csv",
        "models_dir": "../models/sentiment",
        "test_size":  0.4,
        "seed":       1234,
        "average":    "macro",
        "stratified": False,
        "labels":     ["NEGATIVE", "POSITIVE", "NEUTRAL"],
        "preprocess": None,
    },
    "type": {
        "data":       "../data/annotated/human_labelled_type.csv",
        "models_dir": "../models/type",
        "test_size":  0.4,
        "seed":       42,
        "average":    "macro",
        "stratified": True,
        "labels":     ["CITIZENS", "MILITARY", "LOWER", "PEOPLE", "COLONIAL", "FACELESS"],
        "preprocess": lambda df: (
            df[df["label"] != 2]
            .assign(label=lambda d: d["label"].map({0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5}))
            .reset_index(drop=True)
        ),
    },
    "situation": {
        "data":       "../data/annotated/human_labelled_situation.csv",
        "models_dir": "../models/situation",
        "test_size":  0.4,
        "seed":       42,
        "average":    "macro",
        "stratified": True,
        "labels":     ["POLITICS", "ECONOMY", "RELIGION", "SOCIAL", "MILITARY"],
        "preprocess": None,
    },
    "crowd": {
        "data":       "../data/annotated/crowds_df2.csv",
        "models_dir": "../models/crowd",
        "test_size":  0.4,
        "seed":       42,
        "average":    "binary",
        "stratified": False,
        "labels":     ["ABSTRACT", "CROWD"],
        "preprocess": None,
    },
}


def evaluate_models(task: str, models_dir: str = None):
    cfg = TASK_CONFIGS[task]
    models_dir = models_dir or cfg["models_dir"]

    os.makedirs(IMAGES_DIR, exist_ok=True)

    df = pd.read_csv(cfg["data"])
    if cfg["preprocess"]:
        df = cfg["preprocess"](df)
    if cfg["stratified"]:
        split = stratified_split(df, test_size=cfg["test_size"], seed=cfg["seed"])
        test_dataset = split["test"]
    else:
        _, test_df = train_test_split(df, test_size=cfg["test_size"], random_state=cfg["seed"])
        test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg = cfg["average"]

    model_dirs = [
        os.path.join(models_dir, d)
        for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d))
    ]

    results = {}

    for model_path in model_dirs:
        print(f"Evaluating {model_path} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        model.eval()

        preds, labels = [], []
        with torch.no_grad():
            for example in test_dataset:
                inputs = tokenizer(
                    example["text"],
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ).to(device)
                outputs = model(**inputs)
                preds.append(outputs.logits.argmax(-1).item())
                labels.append(example["label"])

        results[os.path.basename(model_path)] = {
            "F1":        f1_score(labels, preds, average=avg),
            "Accuracy":  accuracy_score(labels, preds),
            "Precision": precision_score(labels, preds, average=avg),
            "Recall":    recall_score(labels, preds, average=avg),
            "_preds":    preds,
            "_labels":   labels,
        }

    label_names = cfg.get("labels")

    print(f"\nEvaluation Results ({task}):")
    for model_name, metrics in results.items():
        display = {k: v for k, v in metrics.items() if not k.startswith("_")}
        print(f"  {model_name}: {display}")

    _plot_results(results, task=task)
    _plot_confusion_matrices(results, task=task, label_names=label_names)
    _export_metrics_csv(results, task=task)
    _export_confusion_csv(results, task=task, label_names=label_names)

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_results(results: dict, task: str):
    clean = {m: {k: v for k, v in metrics.items() if not k.startswith("_")}
             for m, metrics in results.items()}
    model_names  = list(clean.keys())
    metric_names = ["F1", "Accuracy", "Precision", "Recall"]
    scores       = np.array([[clean[m][k] for k in metric_names] for m in model_names])

    x         = np.arange(len(model_names))
    bar_width = 0.2

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metric_names):
        plt.bar(x + i * bar_width, scores[:, i], width=bar_width, label=metric)

    plt.xticks(x + bar_width * (len(metric_names) - 1) / 2, model_names, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title(f"Model Comparison — {task}")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(IMAGES_DIR, f"metrics_{task}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved metrics plot    → {fname}")


def _plot_confusion_matrices(results: dict, task: str, label_names: list = None):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4 + (len(label_names or []) // 4)))

    if n == 1:
        axes = [axes]

    for ax, (model_name, metrics) in zip(axes, results.items()):
        cm = confusion_matrix(metrics["_labels"], metrics["_preds"])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ticks = np.arange(len(cm))
        tick_labels = label_names if label_names else [str(i) for i in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)

        thresh = cm.max() / 2
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=9)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(model_name, fontsize=9)

    fig.suptitle(f"Confusion Matrices — {task}", fontsize=11)
    plt.tight_layout()

    fname = os.path.join(IMAGES_DIR, f"confusion_matrices_{task}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved confusion plot  → {fname}")


# ---------------------------------------------------------------------------
# CSV exports for R
# ---------------------------------------------------------------------------

def _export_metrics_csv(results: dict, task: str):
    """
    Tidy long-format metrics table. In R/ggplot2:

        df <- read.csv("images/metrics_human.csv")
        ggplot(df, aes(x=model, y=value, fill=metric)) +
            geom_col(position="dodge") +
            facet_wrap(~task)
    """
    rows = []
    for model_name, metrics in results.items():
        for metric in ("F1", "Accuracy", "Precision", "Recall"):
            rows.append({
                "task":   task,
                "model":  model_name,
                "metric": metric,
                "value":  round(metrics[metric], 6),
            })

    fname = os.path.join(IMAGES_DIR, f"metrics_{task}.csv")
    pd.DataFrame(rows).to_csv(fname, index=False)
    print(f"Saved metrics CSV     → {fname}")


def _export_confusion_csv(results: dict, task: str, label_names: list = None):
    """
    Tidy long-format confusion matrix. In R/ggplot2:

        df <- read.csv("images/confusion_human.csv")
        ggplot(df, aes(x=predicted, y=true, fill=count)) +
            geom_tile() +
            geom_text(aes(label=count)) +
            facet_wrap(~model) +
            scale_fill_gradient(low="white", high="steelblue")
    """
    rows = []
    for model_name, metrics in results.items():
        cm = confusion_matrix(metrics["_labels"], metrics["_preds"])
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                true_lbl = label_names[i] if label_names else str(i)
                pred_lbl = label_names[j] if label_names else str(j)
                rows.append({
                    "task":      task,
                    "model":     model_name,
                    "true":      true_lbl,
                    "predicted": pred_lbl,
                    "count":     int(cm[i, j]),
                })

    fname = os.path.join(IMAGES_DIR, f"confusion_{task}.csv")
    pd.DataFrame(rows).to_csv(fname, index=False)
    print(f"Saved confusion CSV   → {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models")
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--models-dir", default=None,
                        help="Override the default models directory for this task")
    args = parser.parse_args()
    evaluate_models(args.task, args.models_dir)