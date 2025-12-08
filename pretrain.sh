#!/bin/bash
#SBATCH --job-name=trm
#SBATCH --output=out/%x-y-%j.log            # Stdout log file (%x=job name, %j=job ID)
#SBATCH --error=out/%x-y-%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate trp
python pretrain.py --config-name=cfg_sudoku_mlp