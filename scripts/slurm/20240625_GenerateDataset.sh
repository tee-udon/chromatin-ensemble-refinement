#!/bin/bash
#SBATCH --job-name=GenerateDataset
#SBATCH --output=generate_dataset_%j.out
#SBATCH --error=generate_dataset_%j.err
#SBATCH --nodes=1
#SBATCH --partition=ccm 
#SBATCH --mem=1TB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tudomlumleart@flatironinstitute.org

# Load any necessary modules
module load python 

# Activate the venv
source /mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jupyter-gpu-openmm/bin/activate

# Run the script
python /mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/20240626_RunGenerateDataset.py
