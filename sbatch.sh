#!/bin/bash
#SBATCH --job-name=trainingfbi
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --account=ai4bio2025
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=4:00:00

# Carico anaconda (di solito già caricato, ma per sicurezza)
module load anaconda3/2023.09-0-none-none

# Attivo l'env
source activate fbi

# Mi sposto nella cartella del progetto
cd /homes/lpaladino/OS-PREDICTION-WITH-WSI-AND-CLINICAL-DATA

# Rendo la root del progetto visibile a Python
export PYTHONPATH=/homes/lpaladino/OS-PREDICTION-WITH-WSI-AND-CLINICAL-DATA:$PYTHONPATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Creo la cartella logs se non esiste
mkdir -p logs


# ------------------------------
# Lancio training 
# ------------------------------
python root/del/file/da/lanciare