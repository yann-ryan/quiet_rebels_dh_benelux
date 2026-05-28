"""
Stage 2c: Fine-tune situation classifiers on human-labelled data.

The original notebook defined two label sets; the second (3-class) overwrites
the first (5-class). The 3-class set is used here and num_labels is fixed to
match (the notebook had num_labels=5 while only 3 labels were active — a bug).

Input:  data/annotated/human_labelled_situation.csv
        Required columns: text (str), label (int: 0=POLITICS, 1=SOCIAL, 2=MILITARY)

Output: models/situation/<model-name>/
"""

import argparse
import pandas as pd
from train import train_model, cross_validate, stratified_split

USE_CLASS_WEIGHTS = True
EARLY_STOPPING    = 3

ID2LABEL = {0: "POLITICS", 1: "ECONOMY", 2: "RELIGION", 3: "SOCIAL", 4: "MILITARY"}
LABEL2ID = {"POLITICS": 0, "ECONOMY": 1, "RELIGION": 2, "SOCIAL": 3, "MILITARY": 4}

# (checkpoint, learning_rate, epochs, bf16, grad_accum_steps)
MODEL_CONFIGS = [
    ("dbmdz/bert-base-historic-dutch-cased", 2e-5, 10, False, 1),
]


def main(data_path: str, output_root: str, test_size: float, seed: int, cv_folds: int):
    df = pd.read_csv(data_path)
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
    parser = argparse.ArgumentParser(description="Fine-tune situation classifiers")
    parser.add_argument("--data",        default="../data/annotated/human_labelled_situation.csv")
    parser.add_argument("--output-root", default="../models/situation")
    parser.add_argument("--test-size",   type=float, default=0.4)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--cv-folds",    type=int,   default=5,
                        help="Number of CV folds (1 = skip CV, just train)")
    args = parser.parse_args()
    main(args.data, args.output_root, args.test_size, args.seed, args.cv_folds)
