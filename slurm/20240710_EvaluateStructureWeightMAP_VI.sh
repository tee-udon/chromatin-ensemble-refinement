#!/bin/bash
#SBATCH --job-name=EvaluateStructureWeightMAP_VI
#SBATCH --output=/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/slurm/outputs_20240711/EvaluateStructureWeightMAP_%j.out
#SBATCH --error=/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/slurm/outputs_20240711/EvaluateStructureWeightMAP_%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=180G
#SBATCH --ntasks=1

# Load any necessary modules
module load modules/2.2-20230808 cuda/11.8 cudnn/8.9 python/3.11.2

# Activate the venv
source /mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jax_cuda118/bin/activate

# Run the script
python /mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/20240710_EvaluateStructureWeightsMAP_VariedInput.py "$1" "$2"
