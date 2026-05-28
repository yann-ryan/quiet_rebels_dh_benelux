"""
Stage 2b: Fine-tune crowd-type classifiers on human-labelled data.

The original notebook defined two label sets; the second (3-class) overwrites
the first (7-class). The 3-class set is used here and num_labels is fixed to
match (the notebook had num_labels=7 while only 3 labels were active — a bug).

Input:  data/annotated/human_labelled_type.csv
        Required columns: text (str), label (int: 0=CITIZENS, 1=PEOPLE, 2=FACELESS)

Output: models/type/<model-name>/

Original label taxonomy (collapsed to 3):
    CITIZENS, MILITARY, CHILDREN, LOWER, PEOPLE, COLONIAL, FACELESS → CITIZENS / PEOPLE / FACELESS
"""

import argparse
import pandas as pd
from train import train_model, cross_validate, stratified_split

USE_CLASS_WEIGHTS = True
EARLY_STOPPING    = 3

ID2LABEL = {0: "CITIZENS", 1: "MILITARY", 2: "LOWER", 3: "PEOPLE", 4: "COLONIAL", 5: "FACELESS"}
LABEL2ID = {"CITIZENS": 0, "MILITARY": 1, "LOWER": 2, "PEOPLE": 3, "COLONIAL": 4, "FACELESS": 5}

# Original label 2 = CHILDREN (only 2 samples) is dropped.
# Remaining original labels 0,1,3,4,5,6 are remapped to contiguous 0–5.
LABEL_REMAP = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5}

# (checkpoint, learning_rate, epochs, bf16, grad_accum_steps)
MODEL_CONFIGS = [
    ("dbmdz/bert-base-historic-dutch-cased", 2e-5, 10, False, 1),
]


def main(data_path: str, output_root: str, test_size: float, seed: int, cv_folds: int):
    df = pd.read_csv(data_path)
    df = df[df["label"] != 2].copy()                    # drop CHILDREN (2 samples)
    df["label"] = df["label"].map(LABEL_REMAP)
    df = df.reset_index(drop=True)
    split = stratified_split(df, test_size=test_size, seed=seed)

    for checkpoint, lr, epochs, bf16, grad_accum in MODEL_CONFIGS:
        short_name = checkpoint.split("/")[-1]
        output_dir = f"{output_root}/{short_name}"

        if cv_folds > 1:
            print(f"\n=== Cross-validating {checkpoint} ({cv_folds} folds) ===")
            cross_validate(
                df=df,
                model_checkpoint=checkpoint,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                n_folds=cv_folds,
                learning_rate=lr,
                num_train_epochs=epochs,
                bf16=bf16,
                gradient_accumulation_steps=grad_accum,
                early_stopping_patience=EARLY_STOPPING,
                use_class_weights=USE_CLASS_WEIGHTS,
            )

        print(f"\n=== Training {checkpoint} → {output_dir} ===")
        train_model(
            model_checkpoint=checkpoint,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            output_dir=output_dir,
            learning_rate=lr,
            num_train_epochs=epochs,
            bf16=bf16,
            gradient_accumulation_steps=grad_accum,
            early_stopping_patience=EARLY_STOPPING,
            use_class_weights=USE_CLASS_WEIGHTS,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune crowd-type classifiers")
    parser.add_argument("--data",        default="../data/annotated/human_labelled_type.csv")
    parser.add_argument("--output-root", default="../models/type")
    parser.add_argument("--test-size",   type=float, default=0.4)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--cv-folds",    type=int,   default=5,
                        help="Number of CV folds (1 = skip CV, just train)")
    args = parser.parse_args()
    main(args.data, args.output_root, args.test_size, args.seed, args.cv_folds)
