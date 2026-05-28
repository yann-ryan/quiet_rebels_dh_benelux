"""
Stage 4: Run inference on the full Delpher corpus.

Applies four fine-tuned classifiers in sequence:
    1. human     — NONHUMAN / HUMAN
    2. situation — POLITICS / ECONOMY / RELIGION / SOCIAL / MILITARY
    3. type      — CITIZENS / MILITARY / LOWER / PEOPLE / COLONIAL / FACELESS
    4. crowd     — ABSTRACT / CROWD

Input:  data/contexts/*.csv   (each CSV must have a 'Context' column)
Output: data/output/all_massa_inference.csv

Usage:
    python inference.py
    python inference.py --contexts-dir ../data/contexts --output ../data/output/all_massa_inference.csv
"""

import argparse
import glob
import os

import torch
import pandas as pd
from datasets import Dataset
from transformers import pipeline


DEFAULT_MODEL_PATHS = {
    "human":     "../models/human/bert-base-historic-dutch-cased",
    "situation": "../models/situation/bert-base-historic-dutch-cased",
    "type":      "../models/type/bert-base-historic-dutch-cased",
    "crowd":     "../models/crowd/bert-base-historic-dutch-cased",
}


def run_classifier(df: pd.DataFrame, model_path: str, label_col: str, confidence_col: str) -> pd.DataFrame:
    """Apply a single text-classification pipeline to df['Context'] in-place."""
    classifier = pipeline(
        "text-classification",
        model=model_path,
        device=0 if torch.cuda.is_available() else -1,
    )

    # Ensure Context is always a plain string — NaN or None will cause tokenizer errors
    texts = df["Context"].fillna("").astype(str).tolist()

    results = classifier(texts, batch_size=16, truncation=True)
    df[label_col]      = [r["label"] for r in results]
    df[confidence_col] = [r["score"] for r in results]
    return df


def main(contexts_dir: str, output_path: str, model_paths: dict):
    # Load all context CSVs
    csv_files = glob.glob(os.path.join(contexts_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {contexts_dir}")
    print(f"Found {len(csv_files)} CSV file(s)")

    df = pd.concat(
        (pd.read_csv(f) for f in csv_files),
        ignore_index=True,
    )
    print(f"Total rows: {len(df)}")

    classifiers = [
        ("human",     "label_human",     "confidence_human"),
        ("situation", "label_situation",  "confidence_situation"),
        ("type",      "label_type",       "confidence_type"),
        ("crowd",     "label_crowd",      "confidence_crowd"),
    ]

    for key, label_col, conf_col in classifiers:
        model_path = model_paths[key]
        print(f"\nRunning {key} classifier from {model_path} ...")
        df = run_classifier(df, model_path, label_col, conf_col)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on Delpher corpus")
    parser.add_argument("--contexts-dir", default="../data/contexts")
    parser.add_argument("--output", default="../data/output/all_massa_inference.csv")
    parser.add_argument("--human-model",     default=DEFAULT_MODEL_PATHS["human"])
    parser.add_argument("--situation-model", default=DEFAULT_MODEL_PATHS["situation"])
    parser.add_argument("--type-model",      default=DEFAULT_MODEL_PATHS["type"])
    parser.add_argument("--crowd-model",     default=DEFAULT_MODEL_PATHS["crowd"])
    args = parser.parse_args()

    model_paths = {
        "human":     args.human_model,
        "situation": args.situation_model,
        "type":      args.type_model,
        "crowd":     args.crowd_model,
    }
    main(args.contexts_dir, args.output, model_paths)
