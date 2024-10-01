#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH -c 10
#SBATCH --mem=180G

module load python cuda 

source /mnt/home/tudomlumleart/00_VirtualEnvironments/jupyter-gpu-openmm/bin/activate

python3 /mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/20240730_ModelSox9Dataset.py "$1"