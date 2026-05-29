"""
Stage 1: Fine-tune human / non-human classifiers.

Input:  data/annotated/massa_labels2.csv
        Required columns: text (str), label (int: 0=NONHUMAN, 1=HUMAN, 2=AMBIGUOUS)
        Rows with label=2 (AMBIGUOUS) are dropped before training.

Output: models/human/<model-name>/
"""

import argparse
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from train import train_model, cross_validate

USE_CLASS_WEIGHTS     = True
EARLY_STOPPING        = 3

ID2LABEL = {0: "NONHUMAN", 1: "HUMAN"}
LABEL2ID = {"NONHUMAN": 0, "HUMAN": 1}

# (checkpoint, learning_rate, epochs, bf16, grad_accum_steps)
MODEL_CONFIGS = [
    ("dbmdz/bert-base-historic-dutch-cased",        2e-5, 5,  False, 1),
    ("pdelobelle/robbert-v2-dutch-base",             2e-5, 5,  False, 1),
    ("google-bert/bert-base-multilingual-uncased",   2e-5, 5,  False, 1),
    ("intfloat/multilingual-e5-base",                2e-5, 5,  True,  4),
]


def main(data_path: str, output_root: str, test_size: float, seed: int, cv_folds: int):
    df = pd.read_csv(data_path)
    df = df[df["label"] != 2].reset_index(drop=True)   # drop AMBIGUOUS
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

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
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
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
    parser = argparse.ArgumentParser(description="Fine-tune human/non-human classifiers")
    parser.add_argument("--data",        default="../data/annotated/massa_labels2.csv")
    parser.add_argument("--output-root", default="../models/human")
    parser.add_argument("--test-size",   type=float, default=0.4)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--cv-folds",    type=int,   default=5,
                        help="Number of CV folds (1 = skip CV, just train)")
    args = parser.parse_args()
    main(args.data, args.output_root, args.test_size, args.seed, args.cv_folds)
