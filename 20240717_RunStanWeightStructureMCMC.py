# Load necessary modules and dataset 
import sys
sys.path.append(r"/mnt/ceph/users/tudomlumleart/00_VirtualEnvironments/teeu/lib/python3.10/site-packages")
sys.path.append(r"/mnt/home/tudomlumleart/.local/lib/python3.10/site-packages/")
sys.path.append(r"/mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jupyter-gpu/lib/python3.10/site-packages")

import matplotlib.pyplot as plt
from functions import *
from utils import *

import json
import multiprocessing
from cmdstanpy import CmdStanModel

def main():
    my_model = CmdStanModel(
        stan_file='/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/stan/20240715_WeightStructureOptimization.stan',
        cpp_options = {
            "STAN_THREADS": True,
        }
    )
    
    n_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {n_cores}")
    parallel_chains = 4
    threads_per_chain = int(n_cores / parallel_chains)
    print(f"Number of threads per chain: {threads_per_chain}")
    
    output_dir = '/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/MCMC_results/test/20240717_StructureWeight_ChiSq'
    os.makedirs(output_dir, exist_ok=True)
    
    json_filename = '/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/MCMC_results/test/20240716_StructureWeight/data.json'
    stan_output_file = os.path.join(output_dir, 'stan_output')
    
    # Run Stan model to perform MCMC sampling
    data_file = json_filename   
    
    fit = my_model.sample(
        data=data_file,
        chains=4,
        sig_figs=8,
        parallel_chains=parallel_chains,
        threads_per_chain=threads_per_chain,
        iter_warmup=1000,
        iter_sampling=1000,
        show_console=True,
    )
    
    # Save Stan output, i.e., posterior samples, in CSV format, in a specified folder
    fit.save_csvfiles(dir=stan_output_file)

if __name__ == '__main__':
    main()
    