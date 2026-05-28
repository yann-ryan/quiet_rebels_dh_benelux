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

python fine_tune_human.py --cv-folds 1
python fine_tune_sentiment.py --cv-folds 1
python fine_tune_type.py --cv-folds 1
python fine_tune_situation.py --cv-folds 1
python fine_tune_crowd.py --cv-folds 1