"""
Evaluate all fine-tuned models in a given directory against a labelled test set.

Usage:
    python evaluate.py --task human
    python evaluate.py --task crowd
    python evaluate.py --task sentiment
    python evaluate.py --task type
    python evaluate.py --task situation
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import evaluate as hf_evaluate
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train import stratified_split


TASK_CONFIGS = {
    "human": {
        "data":       "data/annotated/massa_labels2.csv",
        "models_dir": "models/human",
        "test_size":  0.4,
        "seed":       42,
        "average":    "macro",
        "stratified": False,
    },
    "sentiment": {
        "data":       "data/annotated/human_labelled_sentiment.csv",
        "models_dir": "models/sentiment",
        "test_size":  0.4,
        "seed":       1234,
        "average":    "macro",
        "stratified": False,
    },
    "type": {
        "data":       "data/annotated/human_labelled_type.csv",
        "models_dir": "models/type",
        "test_size":  0.4,
        "seed":       42,
        "average":    "macro",
        "stratified": True,
    },
    "situation": {
        "data":       "data/annotated/human_labelled_situation.csv",
        "models_dir": "models/situation",
        "test_size":  0.4,
        "seed":       42,
        "average":    "macro",
        "stratified": True,
    },
    "crowd": {
        "data":       "data/annotated/crowds_df2.csv",
        "models_dir": "models/crowd",
        "test_size":  0.4,
        "seed":       42,
        "average":    "binary",
        "stratified": False,
    },
}


def evaluate_models(task: str, models_dir: str = None):
    cfg = TASK_CONFIGS[task]
    models_dir = models_dir or cfg["models_dir"]

    df = pd.read_csv(cfg["data"])
    if cfg["stratified"]:
        split = stratified_split(df, test_size=cfg["test_size"], seed=cfg["seed"])
    else:
        dataset = Dataset.from_pandas(df)
        split = dataset.train_test_split(test_size=cfg["test_size"], seed=cfg["seed"])
    test_dataset = split["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg = cfg["average"]

    f1_metric        = hf_evaluate.load("f1")
    accuracy_metric  = hf_evaluate.load("accuracy")
    precision_metric = hf_evaluate.load("precision")
    recall_metric    = hf_evaluate.load("recall")

    model_dirs = [
        os.path.join(models_dir, d)
        for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d))
    ]

    results = {}

    for model_path in model_dirs:
        print(f"Evaluating {model_path} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
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
            "F1":        f1_metric.compute(predictions=preds, references=labels, average=avg)["f1"],
            "Accuracy":  accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
            "Precision": precision_metric.compute(predictions=preds, references=labels, average=avg)["precision"],
            "Recall":    recall_metric.compute(predictions=preds, references=labels, average=avg)["recall"],
        }

    print(f"\nEvaluation Results ({task}):")
    for model_name, metrics in results.items():
        print(f"  {model_name}: {metrics}")

    _plot_results(results, title=f"Model Comparison — {task}")
    return results


def _plot_results(results: dict, title: str):
    model_names  = list(results.keys())
    metric_names = ["F1", "Accuracy", "Precision", "Recall"]
    scores       = np.array([[results[m][k] for k in metric_names] for m in model_names])

    x         = np.arange(len(model_names))
    bar_width = 0.2

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metric_names):
        plt.bar(x + i * bar_width, scores[:, i], width=bar_width, label=metric)

    plt.xticks(x + bar_width * (len(metric_names) - 1) / 2, model_names, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').replace('—', '-')}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models")
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--models-dir", default=None,
                        help="Override the default models directory for this task")
    args = parser.parse_args()
    evaluate_models(args.task, args.models_dir)
