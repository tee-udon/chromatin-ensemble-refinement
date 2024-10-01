# Load necessary modules and dataset 
import sys
sys.path.append(r"/mnt/ceph/users/tudomlumleart/00_VirtualEnvironments/teeu/lib/python3.10/site-packages")
sys.path.append(r"/mnt/home/tudomlumleart/.local/lib/python3.10/site-packages/")
sys.path.append(r"/mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jupyter-gpu/lib/python3.10/site-packages")

import matplotlib.pyplot as plt
from functions import *
from utils import *

import torch
import json
import multiprocessing
from cmdstanpy import CmdStanModel

def logprior(dmap_flat, num_probes):
    """
    """
    return np.sum(np.array(logprior_(dmap_flat, num_probes)))

def logprior_(dmap_flat, num_probes):
    """
    """
    # Get 2D map back to simplify the expression 
    dmap = np.reshape(dmap_flat, [num_probes, num_probes])
    
    # Calculate the squared end-to-end distance 
    R_sq = dmap[0][-1] ** 2
    
    # Calculate the average bond length
    b = np.mean(np.diag(dmap, 1))
    
    N = num_probes
    
    # Calculate the probability
    scaling_factor = 1.5 * np.log(3/(2*np.pi*N*b**2))
    gaussian_term = -3*R_sq/(2*N*b**2)
    
    return scaling_factor, gaussian_term 


def loglikelihood(dmap_flat, ref_dmap_flat, measurement_error, num_probes):
    """
    """
    return np.sum(np.array(loglikelihood_(dmap_flat, ref_dmap_flat, measurement_error, num_probes)))


def loglikelihood_(dmap_flat, ref_dmap_flat, measurement_error, num_probes):
    """ 
    """
    # Calculate the difference between distance map and reference 
    # distance map
    subtraction_map_sq = np.square(dmap_flat - ref_dmap_flat)
    sum_subtraction_map_sq = np.sum(subtraction_map_sq)
    
    # Calculate the normalization factor
    normalization_factor = -np.square(num_probes) * np.log(np.sqrt(2*np.pi*np.square(measurement_error)))
    
    # Calculate the gaussian term 
    gaussian_term = -np.sum(sum_subtraction_map_sq)/(2*np.square(measurement_error))
    
    return normalization_factor, gaussian_term

def run_mcmc(dataset_list, param_dict, save_dir, dataset_idx):
    # Unpack the param dict 
    num_monomers = param_dict['num_monomers']
    mean_bond_length = param_dict['mean_bond_length']
    std_bond_length = param_dict['std_bond_length'] 
    num_templates = param_dict['num_templates']
    measurement_error = param_dict['noise_std']
    weight_dist = param_dict['weights_dist']
    num_observations = param_dict['num_observations']
    num_probes = num_monomers
    num_candidates = num_templates
    
    # Generate variables for the optimization
    template_list = dataset_list[dataset_idx]['template_chain_list']
    X = dataset_list[dataset_idx]['observation_list'][:num_observations]
    label_list = dataset_list[dataset_idx]['labels']
    observation_flatten_list = [squareform(pdist(x)).flatten() for x in X]
    
    # generate weight of each label from label_list
    true_weights = np.array([np.sum(label_list == i) for i in np.unique(label_list)]) / len(label_list)
    true_weights = true_weights.reshape(-1, 1)
    templates_flatten = [squareform(pdist(x)).flatten() for x in template_list]
    
    # Generate log prior for metastructures 
    lpm = [(logprior(x, num_monomers)).tolist() for x in templates_flatten]
    
    # Generate log likelihood for observations given metastructures 
    ll = [[(loglikelihood(y, x, measurement_error, num_monomers)).tolist() for x in templates_flatten] for y in observation_flatten_list]
    
    N = num_observations
    M = num_candidates
    
    # Load stan model 
    my_model = CmdStanModel(
        stan_file='/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/stan/20240715_WeightOptimization.stan',
        cpp_options = {
            "STAN_THREADS": True,
        }
        )
    
    n_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {n_cores}")
    parallel_chains = 4
    threads_per_chain = int(n_cores / parallel_chains)
    print(f"Number of threads per chain: {threads_per_chain}")
    
    output_dir = os.path.join(save_dir, str(dataset_idx))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Write json files for reading into stan program
    json_filename = os.path.join(output_dir, 'data.json')
    stan_output_file = os.path.join(output_dir, 'stan_output')
    data_dict = {
        "M": M,
        "N": N,
        "ll_map": ll,
        "lpm_vec": lpm,
    }

    json_obj = json.dumps(data_dict, indent=4)

    with open(json_filename, 'w') as json_file:
        json_file.write(json_obj)
        json_file.close()
        
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
    
    # Write true weights to the output directory
    np.savetxt(os.path.join(output_dir, "true_weights.txt"), true_weights.T, fmt='%.6f')
    
    
# Run file from the command line 
if __name__ == '__main__':
    # Accept the path to a pickle file as an argument
    p = sys.argv[1]
    
    # Load the dataset from that pickle file 
    print("Loading dataset...")
    dataset_list, param_dict = load_dataset(p)
    
    # Generate save directory for the result 
    # Extracting the core name of the pickle file
    core_name = p.split("/")[-1].split(".")[0]
    mcmc_result_dir = '/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/MCMC_results'
    save_dir = os.path.join(mcmc_result_dir, core_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Run the MCMC for 100 datasets in the list
    for i in tqdm(range(100)):
        print(f"Running MCMC for dataset {i}...")
        run_mcmc(dataset_list, param_dict, save_dir, i)
    
    