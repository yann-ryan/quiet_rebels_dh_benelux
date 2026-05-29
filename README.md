# Code and training data for the conference paper Using Large Language Models to understand ‘the masses’ in nineteenth-century Dutch newspapers 

Fine-tuning pipeline for detecting human references, crowd mentions, sentiment,
crowd type, and situation Delpher, presented at DH Benelux 2026. 

[add zenodo doi here]

---

## Pipeline overview

```
Stage 1   Select 1000 examples at random from 1870-1880. Word 'massa' and a context window of 40 tokens on either side. (`massa_labels2.csv`)

Stage 1   Annotate `massa_labels2.csv` as human/non-human/ambiguous sense
          → fine-tune human/non-human models (AMBIGUOUS rows dropped). 
    
Stage 3   Use model to generate predictions for all of 1870-1880 corpus, select 300 high-confidence human sense. This is used for the `human_labelled_` training datasets. 

Stage 2a  Annotate human_labelled_sentiment.csv
          → fine-tune sentiment models

Stage 2b  Annotate human_labelled_type.csv
          → fine-tune crowd-type models

Stage 2c  Annotate human_labelled_situation.csv
          → fine-tune situation models

Stage 3   Run human/non-human inference on full Delpher corpus
          → sample 2000 HUMAN results, remove advertisements
          → annotate crowds_df2.csv
          → fine-tune crowd/abstract models

Stage 4   Run all five classifiers on the full Delpher corpus
          → data/output/all_massa_inference.csv
```

---

## Repository structure

```
quiet-rebels-sentiment/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── annotated/                          # manually labelled CSVs (track in git)
│   │   ├── massa_labels2.csv               # stage 1
│   │   ├── human_labelled_sentiment.csv    # stage 2a
│   │   ├── human_labelled_type.csv         # stage 2b
│   │   ├── human_labelled_situation.csv    # stage 2c
│   │   └── crowds_df2.csv                  # stage 3
│   ├── contexts/                           # raw Delpher context CSVs (gitignored)
│   │   └── *.csv
│   └── output/                             # inference results (gitignored)
│       └── all_massa_inference.csv
│
├── models/                                 # saved fine-tuned models (gitignored)
│   ├── human/
│   ├── sentiment/
│   ├── type/
│   ├── situation/
│   └── crowd/
│
└── src/
    ├── train.py                  # shared utilities: train_model, cross_validate, stratified_split
    ├── fine_tune_human.py        # stage 1
    ├── fine_tune_sentiment.py    # stage 2a
    ├── fine_tune_type.py         # stage 2b
    ├── fine_tune_situation.py    # stage 2c
    ├── fine_tune_crowd.py        # stage 3
    ├── evaluate_models.py        # evaluate any task against its held-out test set
    └── inference.py              # stage 4 — full Delpher inference
```

---

## Data files

| File | Stage | Columns | Labels |
|---|---|---|---|
| `massa_labels2.csv` | 1 | `text`, `label` | 0=NONHUMAN, 1=HUMAN *(label 2=AMBIGUOUS dropped)* |
| `human_labelled_sentiment.csv` | 2a | `text`, `label` | 0=NEGATIVE, 1=POSITIVE, 2=NEUTRAL |
| `human_labelled_type.csv` | 2b | `text`, `label` | 0=CITIZENS, 1=PEOPLE, 2=FACELESS |
| `human_labelled_situation.csv` | 2c | `text`, `label` | 0=POLITICS, 1=SOCIAL, 2=MILITARY |
| `crowds_df2.csv` | 3 | `text`, `label` | 0=ABSTRACT, 1=CROWD |
| `data/contexts/*.csv` | 4 | `Context` | — (inference input) |

All train/test splits use **test_size=0.4**. Type and situation use stratified
splits to handle class imbalance.

---

## Models trained per task

| Task | Models |
|---|---|
| human | bert-base-historic-dutch-cased, robbert-v2-dutch-base, bert-base-multilingual-uncased, multilingual-e5-base |
| sentiment | same 4 + nlptown/bert-base-multilingual-uncased-sentiment |
| type | bert-base-historic-dutch-cased |
| situation | bert-base-historic-dutch-cased |
| crowd | bert-base-historic-dutch-cased, robbert-v2-dutch-base, bert-base-multilingual-uncased, multilingual-e5-base |

---

## Training approach

All fine-tuning runs through `train.py` which provides:

- **Weighted cross-entropy loss** — class weights computed via `sklearn`'s
  `compute_class_weight("balanced", ...)` and applied during training, so
  minority classes receive proportionally higher loss weight.
- **Early stopping** — training halts if macro F1 on the eval set does not
  improve for 3 consecutive epochs; the best checkpoint is restored at the end.
- **F1-based model selection** — `metric_for_best_model="f1"` (macro) rather
  than accuracy, so the saved model is the one most balanced across all classes.
- **Stratified k-fold cross-validation** — each fine-tune script accepts a
  `--cv-folds` argument (default 5). CV runs on the full dataset before final
  model training and reports mean ± std metrics across folds.

---

## Usage

```bash
pip install -r requirements.txt
cd src
```

### Run each training stage

```bash
# Standard run (includes 5-fold CV before each model trains)
python fine_tune_human.py
python fine_tune_sentiment.py
python fine_tune_type.py
python fine_tune_situation.py
python fine_tune_crowd.py

# Skip CV and just train
python fine_tune_human.py --cv-folds 1
```

### Evaluate

```bash
python evaluate_models.py --task human
python evaluate_models.py --task sentiment
python evaluate_models.py --task type
python evaluate_models.py --task situation
python evaluate_models.py --task crowd
```

### Run full Delpher inference (stage 4)

```bash
python inference.py \
  --contexts-dir  ../data/contexts \
  --output        ../data/output/all_massa_inference.csv \
  --human-model     ../models/human/bert-base-historic-dutch-cased \
  --situation-model ../models/situation/bert-base-historic-dutch-cased \
  --type-model      ../models/type/bert-base-historic-dutch-cased \
  --crowd-model     ../models/crowd/bert-base-historic-dutch-cased
```

All default paths can be overridden with `--data`, `--output-root`, `--test-size`, `--seed`.

---

