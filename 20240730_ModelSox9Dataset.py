import sys
sys.path.append(r"/mnt/ceph/users/tudomlumleart/00_VirtualEnvironments/teeu/lib/python3.10/site-packages")
sys.path.append(r"/mnt/home/tudomlumleart/.local/lib/python3.10/site-packages/")
sys.path.append(r"/mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jupyter-gpu/lib/python3.10/site-packages")
from utils import *
from functions import *
from _polychrom import *

def main(dataset):
    column_names = ['chrom', 'chromStart', 'chromEnd', 'name', 'score',
              'strand', 'signalValue', 'pValue', 'qValue', 'peak']

    chrom = 'chr17'
    coord = [68066461, 72018460]
    
    ctcf_esc_path = '/mnt/home/tudomlumleart/ceph/05_Sox9Dataset/ctcf_chip_esc.bed'
    ctcf_cnc_path = '/mnt/home/tudomlumleart/ceph/05_Sox9Dataset/ctcf_chip_cnc.bed'
    
    ctcf_esc_df = pd.read_csv(ctcf_esc_path, sep='\t', header=None)
    ctcf_cnc_df = pd.read_csv(ctcf_cnc_path, sep='\t', header=None)
    ctcf_esc_df.columns = column_names
    ctcf_cnc_df.columns = column_names
    
    condition = lambda df: (df.chrom == chrom) & (df.chromStart >= coord[0]) & (df.chromEnd <= coord[1]) & (df.score == 1000)
    
    ctcf_esc_df = ctcf_esc_df[condition(ctcf_esc_df)]
    ctcf_cnc_df = ctcf_cnc_df[condition(ctcf_cnc_df)]
    
    ctcf_esc_df['loc_bin'] = pd.cut(ctcf_esc_df['chromStart'], bins=np.linspace(coord[0], coord[1], 100), labels=False)
    ctcf_cnc_df['loc_bin'] = pd.cut(ctcf_cnc_df['chromStart'], bins=np.linspace(coord[0], coord[1], 100), labels=False)
    
    def logistic_function(x):
        # this make sure that the boundary between two tads are strong
        mean = x.signalValue[x.signalValue < 1000].median()
        slope = 1e-2
        return 1/(1 + np.exp(-slope*(x.signalValue-mean)))
    
    ctcf_esc_df['BEprob'] = logistic_function(ctcf_esc_df)
    ctcf_cnc_df['BEprob'] = logistic_function(ctcf_cnc_df)
    
    ctcf_esc_df = ctcf_esc_df[ctcf_esc_df.signalValue > ctcf_esc_df.signalValue.quantile(0.75)]
    ctcf_cnc_df = ctcf_esc_df[ctcf_esc_df.signalValue > ctcf_esc_df.signalValue.quantile(0.75)]
    
    save_folder_esc = '/mnt/home/tudomlumleart/ceph/01_TetheringSimulation/LoopExtrusion_Sox9_ESC'
    save_folder_cnc = '/mnt/home/tudomlumleart/ceph/01_TetheringSimulation/LoopExtrusion_Sox9_CNC'
    
    # Test generating polymers
    num_monomers = 100
    num_polymers = 1
    num_observations = 1000 # per template
    
    if dataset == 'ESC':
        ctcf_sites = ctcf_esc_df.loc_bin.values
        ctcf_stall_probs = ctcf_esc_df.BEprob.values 
    
        generate_polymer_chain(num_monomers, num_polymers, 
                        num_observations, save_folder_esc,
                        ctcf_sites=ctcf_sites, ctcf_stall_probs=ctcf_stall_probs,
                        num_templates=1)
    
    elif dataset == 'CNC':
        ctcf_sites = ctcf_cnc_df.loc_bin.values
        ctcf_stall_probs = ctcf_cnc_df.BEprob.values
        
        generate_polymer_chain(num_monomers, num_polymers,
                        num_observations, save_folder_cnc,
                        ctcf_sites=ctcf_sites, ctcf_stall_probs=ctcf_stall_probs,
                        num_templates=1)
            

# Run file from the command line 
if __name__ == '__main__':
    dataset = sys.argv[1]
    main(dataset)