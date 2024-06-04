from functions import *

column_names = ['chrom', 'chromStart', 'chromEnd', 'name', 'score',
              'strand', 'signalValue', 'pValue', 'qValue', 'peak']

chrom = 'chr6'
coord = [52012980, 52345139]

ctcf_chip_path = '/mnt/home/tudomlumleart/ceph/01_TetheringSimulation/genomeData/ctcfChIP/GSE96107_ES_CTCF.IDR0.05.filt.narrowPeak.gz' 
ring_chip_path = '/mnt/home/tudomlumleart/ceph/01_TetheringSimulation/genomeData/ring1ChIP/GSE96107_ES_CTCF.IDR0.05.filt.narrowPeak.gz'
rad21_chip_path = '/mnt/home/tudomlumleart/ceph/01_TetheringSimulation/genomeData/rad21ChIP/GSE137272_Rad21-ChIPseq.narrowPeak'

ring_df = pd.read_csv(ring_chip_path, delimiter='\t', header=None)
ctcf_df = pd.read_csv(ctcf_chip_path, delimiter='\t', header=None)
rad21_df = pd.read_csv(rad21_chip_path, delimiter='\t', header=None, skiprows=1)

ring_df.columns = column_names
ctcf_df.columns = column_names
rad21_df.columns = column_names

condition = lambda df: (df.chrom == chrom) & (df.chromStart >= coord[0]) & (df.chromEnd <= coord[1]) & (df.score == 1000)

ring_df = ring_df.loc[condition(ring_df)]
ctcf_df = ctcf_df.loc[(condition(ctcf_df)) & (ctcf_df.signalValue > ctcf_df.signalValue.median())]
rad21_df = rad21_df.loc[(rad21_df.chrom == chrom) & (rad21_df.chromStart >= coord[0]) & (rad21_df.chromEnd <= coord[1]) & (rad21_df.score > rad21_df.score.median())]

ctcf_df['bin'] = pd.cut(ctcf_df['chromStart'], bins=np.arange(coord[0], coord[1], 1000), labels=False)
ring_df['bin'] = pd.cut(ring_df['chromStart'], bins=np.arange(coord[0], coord[1], 1000), labels=False)
rad21_df['bin'] = pd.cut(rad21_df['chromStart'], bins=np.arange(coord[0], coord[1], 1000), labels=False)

rad21_ctcf_common = set(rad21_df.bin) & set(ctcf_df.bin)

ctcf_df = ctcf_df.loc[ctcf_df.bin.isin(rad21_ctcf_common)]
rad21_df = rad21_df.loc[rad21_df.bin.isin(rad21_ctcf_common)]

# direction is calculated by 
# checking if rad21 peak is to the left of ctcf peak, if True, set it to 2
# else, set it to 1
ctcf_dir = (((rad21_df.chromStart + rad21_df.peak).values - (ctcf_df.chromStart + ctcf_df.peak).values) < 0) + 1

ctcf_df['dir'] = ctcf_dir

def logistic_function(x):
    # this make sure that the boundary between two tads are strong
    mean = x.signalValue[x.signalValue < 1000].median()
    slope = 0.01
    return 1/(1 + np.exp(-slope*(x.signalValue-mean)))

ctcf_df['BEprob'] = logistic_function(ctcf_df)

ring_level_bin = np.linspace(ring_df.signalValue.min()-1, ring_df.signalValue.max(), 4)
ring_df['levelBin'] = pd.cut(ring_df['signalValue'], bins=ring_level_bin, labels=False)

save_folder = '/mnt/home/tudomlumleart/ceph/01_TetheringSimulation/LoopExtrusion_HoxA_mESC_Good1/'

# Test generating polymers
num_monomers = len(np.arange(coord[0], coord[1], 1000))
num_polymers = 1
num_observations = 5000 
ctcf_sites = ctcf_df.bin.values
ctcf_directions = ctcf_df.dir.values
ctcf_stall_probs = ctcf_df.BEprob.values 
monomer_types = np.zeros(num_monomers).astype(int)
monomer_types[ring_df.bin.values] = ring_df.levelBin.values 
interaction_matrix = np.array([[0, 0, 0], [0, 1.5, 1.5], [0, 1.5, 2]])

generate_polymer_chain(num_monomers, num_polymers, 
                       num_observations, save_folder,
                       monomer_types=monomer_types,
                       interaction_matrix=interaction_matrix, 
                       ctcf_sites=ctcf_sites,
                       ctcf_directions=ctcf_directions,
                       ctcf_stall_probs=ctcf_stall_probs)