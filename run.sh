#!/bin/bash
#SBATCH --partition=SCSEGPU_M1
#SBATCH --qos=q_amsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --job-name=MyJob
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load anaconda3/23.5.2
eval "$(conda shell.bash hook)"
conda activate TestEnv
python main.py