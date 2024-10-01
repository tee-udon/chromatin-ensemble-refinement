#!/bin/bash
#SBATCH --job-name=modelFit
#SBATCH --output=/mnt/home/tudomlumleart/ceph/job_outputs/modelFit_%j.out
#SBATCH --error=/mnt/home/tudomlumleart/ceph/job_outputs/modelFit_%j.err
#SBATCH --partition=ccm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1TB

# Load any necessary modules
module load modules/2.2-20230808 python/3.11.2

# Activate the venv
source /mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jax_cuda118/bin/activate

# Run the script
python /mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/20240626_DetermineBGMMFit.py "$1"
