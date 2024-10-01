#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH -c 10
#SBATCH --mem=180G
#SBATCH --mail-user=tudomlumleart@flatironinstitute.org

module load cuda 

source /mnt/home/tudomlumleart/00_VirtualEnvironments/jupyter-gpu-openmm/bin/activate
python3 /mnt/home/tudomlumleart/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/20240604_GenerateHoxASimulations.py