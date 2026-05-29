#!/bin/bash
#SBATCH --job-name=evaluate_all
#SBATCH --mail-user="y.c.ryan@hum.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

module load Python/3.11.5-GCCcore-13.2.0

source /home/ryanyc/data1/venv_choicellm/bin/activate

export HF_HOME=/home/ryanyc/data1/huggingface

pip install -r requirements.txt
cd src

python evaluate_models.py --task human
python evaluate_models.py --task crowd
python evaluate_models.py --task sentiment
python evaluate_models.py --task type
python evaluate_models.py --task situation