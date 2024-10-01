#!/bin/bash
#SBATCH --job-name=WeightMCMC
#SBATCH --output=/mnt/home/tudomlumleart/ceph/job_outputs/MCMC_%j.out
#SBATCH --error=/mnt/home/tudomlumleart/ceph/job_outputs/MCMC_%j.err
#SBATCH --nodes=1
#SBATCH --partition=ccm
#SBATCH --mem=180G
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20

# Load any necessary modules
module load modules/2.2-20230808 python/3.11.2

# Activate the venv
source /mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jax_cuda118/bin/activate

# Run the script
# Now accept any number of arguments
python "$@"
