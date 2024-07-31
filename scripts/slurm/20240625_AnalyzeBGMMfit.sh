#!/bin/bash
#SBATCH --job-name=modelFit
#SBATCH --output=/mnt/home/tudomlumleart/ceph/job_outputs/modelFit_%j.out
#SBATCH --error=/mnt/home/tudomlumleart/ceph/job_outputs/modelFit_%j.err
#SBATCH --partition=ccm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1TB
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tudomlumleart@flatironinstitute.org

module load python

source /mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jupyter-gpu-openmm/bin/activate

python /mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/20240626_DetermineBGMMFit.py "$1"
