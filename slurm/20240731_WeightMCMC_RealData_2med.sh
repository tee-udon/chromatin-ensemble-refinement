#!/bin/bash
#SBATCH --job-name=WeightMCMC
#SBATCH --output=/mnt/home/tudomlumleart/ceph/job_outputs/20240731_MCMC_Weight_RealData_2med/EvaluateStructureWeightMAP_%j.out
#SBATCH --error=/mnt/home/tudomlumleart/ceph/job_outputs/20240731_MCMC_Weight_RealData_2med/EvaluateStructureWeightMAP_%j.err
#SBATCH --nodes=1
#SBATCH --partition=ccm
#SBATCH --mem=180G
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=10

# Load any necessary modules
module load modules/2.2-20230808 python/3.11.2

# Activate the venv
source /mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jax_cuda118/bin/activate

# Run the script
python /mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/20240731_RealData_WeightOpt_2med.ipynb
