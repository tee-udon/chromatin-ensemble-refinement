#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tudomlumleart@flatironinstitute.org

~/ceph/00_Softwares/sequencingFunctions/bigWigToBedGraph  /mnt/home/tudomlumleart/ceph/01_TetheringSimulation/genomeData/rad21ChIP/GSE137272_Rad21-ChIPseq.bw /mnt/home/tudomlumleart/ceph/01_TetheringSimulation/genomeData/rad21ChIP/GSE137272_Rad21-ChIPseq.bedgraph
