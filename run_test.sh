#!/bin/bash
#SBATCH --partition=SCSEGPU_M2
#SBATCH --qos=q_dmsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --job-name=fashion
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=output/error_%x_%j.err

module load anaconda3/23.5.2
eval "$(conda shell.bash hook)"
conda activate mobilenet
python /home/msai/xi0001ye/Fashion_Detect_Project/test.py