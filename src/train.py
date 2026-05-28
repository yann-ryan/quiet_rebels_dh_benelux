"""
Shared fine-tuning utilities.
Each task-specific script calls `train_model`, `cross_validate`, or `stratified_split`.

Changes vs. original:
  - WeightedTrainer: weighted cross-entropy loss for class imbalance
  - compute_metrics: returns macro F1 + accuracy; F1 drives model selection
  - train_model: metric_for_best_model="f1", EarlyStoppingCallback
  - cross_validate: stratified k-fold CV reporting mean ± std metrics
"""

import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "f1":       f1_score(labels, predictions, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, predictions),
    }


# ---------------------------------------------------------------------------
# Weighted Trainer
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """Trainer with optional class-weighted cross-entropy loss."""

    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # 1-D float tensor, one weight per class

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = (
            self.class_weights.to(outputs.logits.device)
            if self.class_weights is not None
            else None
        )
        loss = F.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def _class_weights(dataset, num_labels: int) -> torch.Tensor:
    """Compute balanced class weights from a HuggingFace Dataset.

    Derives classes from the data rather than np.arange(num_labels) to avoid
    a mismatch when stratified_split re-indexes labels via ClassLabel.
    Weights for any class absent from this split default to 1.0.
    """
    y = np.array(dataset["label"])
    present = np.unique(y)
    weights_present = compute_class_weight("balanced", classes=present, y=y)
    full_weights = np.ones(num_labels, dtype=float)
    for cls, w in zip(present, weights_present):
        full_weights[int(cls)] = w
    return torch.tensor(full_weights, dtype=torch.float)


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_model(
    model_checkpoint: str,
    train_dataset,
    eval_dataset,
    id2label: dict,
    label2id: dict,
    output_dir: str,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    weight_decay: float = 0.01,
    bf16: bool = False,
    gradient_accumulation_steps: int = 1,
    early_stopping_patience: int = 3,
    use_class_weights: bool = False,
):
    """Fine-tune a sequence classification model and save to output_dir."""
    num_labels = len(id2label)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True)

    tok_train = train_dataset.map(preprocess, batched=True)
    tok_eval  = eval_dataset.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    weights = _class_weights(train_dataset, num_labels) if use_class_weights else None

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=False,
        bf16=bf16,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        class_weights=weights,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model to {output_dir}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(
    df,
    model_checkpoint: str,
    id2label: dict,
    label2id: dict,
    n_folds: int = 5,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 4,
    weight_decay: float = 0.01,
    bf16: bool = False,
    gradient_accumulation_steps: int = 1,
    early_stopping_patience: int = 3,
    use_class_weights: bool = False,
    tmp_dir: str = "/tmp/cv_folds",
) -> dict:
    """
    Stratified k-fold cross-validation.

    Trains and evaluates `n_folds` models in turn, cleans up each fold's
    checkpoint afterwards, and returns per-fold and averaged metrics.

    Does NOT save a final model — call train_model() separately for that.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    labels = df["label"].values
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
        print(f"\n  [CV {model_checkpoint.split('/')[-1]}] Fold {fold + 1}/{n_folds}")
        fold_dir = os.path.join(tmp_dir, f"fold_{fold}")

        train_ds = Dataset.from_pandas(df.iloc[train_idx].reset_index(drop=True))
        val_ds   = Dataset.from_pandas(df.iloc[val_idx].reset_index(drop=True))

        train_model(
            model_checkpoint=model_checkpoint,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            id2label=id2label,
            label2id=label2id,
            output_dir=fold_dir,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            weight_decay=weight_decay,
            bf16=bf16,
            gradient_accumulation_steps=gradient_accumulation_steps,
            early_stopping_patience=early_stopping_patience,
            use_class_weights=use_class_weights,
        )

        # Evaluate best checkpoint on the held-out fold
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(fold_dir, use_fast=False)
        mdl = AutoModelForSequenceClassification.from_pretrained(fold_dir).to(device)
        mdl.eval()

        preds, true = [], []
        with torch.no_grad():
            for ex in val_ds:
                inp = tok(
                    ex["text"], truncation=True, padding="max_length", return_tensors="pt"
                ).to(device)
                out = mdl(**inp)
                preds.append(out.logits.argmax(-1).item())
                true.append(ex["label"])

        metrics = {
            "f1":        f1_score(true, preds, average="macro", zero_division=0),
            "accuracy":  accuracy_score(true, preds),
            "precision": precision_score(true, preds, average="macro", zero_division=0),
            "recall":    recall_score(true, preds, average="macro", zero_division=0),
        }
        fold_results.append(metrics)
        shutil.rmtree(fold_dir, ignore_errors=True)

        print(
            f"  [CV] Fold {fold + 1} — "
            f"F1: {metrics['f1']:.3f}  Acc: {metrics['accuracy']:.3f}"
        )

    mean = {k: float(np.mean([r[k] for r in fold_results])) for k in fold_results[0]}
    std  = {k: float(np.std( [r[k] for r in fold_results])) for k in fold_results[0]}

    print(f"\n  [CV] {n_folds}-fold mean ± std:")
    for k in mean:
        print(f"    {k:10s}: {mean[k]:.3f} ± {std[k]:.3f}")

    return {"folds": fold_results, "mean": mean, "std": std}


# ---------------------------------------------------------------------------
# Stratified split helper (used by type / situation)
# ---------------------------------------------------------------------------

def stratified_split(df, label_col: str = "label", test_size: float = 0.4, seed: int = 42):
    """
    Build a stratified train/test split from a DataFrame.
    Uses HuggingFace Dataset's stratify_by_column.
    """
    from datasets import ClassLabel

    dataset = Dataset.from_pandas(df)
    unique_labels = sorted(set(dataset[label_col]))
    class_label = ClassLabel(names=[str(l) for l in unique_labels])
    dataset = dataset.cast_column(label_col, class_label)
    return dataset.train_test_split(
        test_size=test_size, seed=seed, stratify_by_column=label_col
    )
