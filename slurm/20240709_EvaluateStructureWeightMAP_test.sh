#!/bin/bash
#SBATCH --job-name=EvaluateStructureWeightMAP
#SBATCH --output=/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/slurm/outputs/EvaluateStructureWeightMAP_%j.out
#SBATCH --error=/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/slurm/outputs/EvaluateStructureWeightMAP_%j.err
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
python /mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/20240708_EvaluateStructuresWeightsMAP.py /mnt/home/tudomlumleart/ceph/03_GaussianChainSimulation/20240627/dataset_100_10_20_500_1000_100.0_10000.pkl
