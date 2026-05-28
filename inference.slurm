#!/bin/bash
#SBATCH --job-name=fine_tune_all
#SBATCH --mail-user="y.c.ryan@hum.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8


module load Python/3.11.5-GCCcore-13.2.0

source /home/ryanyc/data1/venv_choicellm/bin/activate

export HF_HOME=/home/ryanyc/data1/huggingface

pip install -r requirements.txt
cd src

python inference.py \
  --contexts-dir  /data1/ryanyc/quiet_rebels_sentiment/data/input/contexts \
  --output        ../data/output/all_massa_inference.csv \
  --human-model     ../models/human/bert-base-historic-dutch-cased \
  --situation-model ../models/situation/bert-base-historic-dutch-cased \
  --type-model      ../models/type/bert-base-historic-dutch-cased \
  --crowd-model     ../models/crowd/bert-base-historic-dutch-cased