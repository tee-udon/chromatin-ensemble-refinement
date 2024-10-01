#!/bin/bash
#SBATCH --job-name=modelFit
#SBATCH --output=/mnt/home/tudomlumleart/ceph/job_outputs/modelFit_%j.out
#SBATCH --error=/mnt/home/tudomlumleart/ceph/job_outputs/modelFit_%j.err
#SBATCH --partition=ccm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-node=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tudomlumleart@flatironinstitute.org

module load python 

source /mnt/home/tudomlumleart/00_VirtualEnvironments/jupyter-gpu-openmm/bin/activate

python /mnt/home/tudomlumleart/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/20240626_DetermineBGMMFit.py /mnt/home/tudomlumleart/ceph/03_GaussianChainSimulation/20240625/dataset_100_10_20_2_0.1_2.0_10000_202.pkl
