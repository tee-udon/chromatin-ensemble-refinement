#!/bin/bash
#SBATCH --job-name=WeightMCMC
#SBATCH --output=/mnt/home/tudomlumleart/ceph/job_outputs/LoopCaller_%j.out
#SBATCH --error=/mnt/home/tudomlumleart/ceph/job_outputs/LoopCaller_%j.err
#SBATCH --nodes=1
#SBATCH --partition=ccm
#SBATCH --mem=180G
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20

# Load any necessary modules
module load modules/2.2-20230808 python/3.11.2

# Activate the venv
source /mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/teeu/bin/activate

# Run the script
mustache -f "$1" -r "$2" -o "$3" -pt 0.1 -p $(nproc)

