#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tudomlumleart@flatironinstitute.org

source ~/00_VirtualEnvironments/jupyter-gpu-openmm/bin/activate 
macs3 bdgpeakcall -i /mnt/home/tudomlumleart/ceph/01_TetheringSimulation/genomeData/rad21ChIP/GSE137272_Rad21-ChIPseq.bedgraph -o /mnt/home/tudomlumleart/ceph/01_TetheringSimulation/genomeData/rad21ChIP/GSE137272_Rad21-ChIPseq.narrowPeak