# Load necessary modules and dataset 
import sys
sys.path.append(r"/mnt/ceph/users/tudomlumleart/00_VirtualEnvironments/teeu/lib/python3.10/site-packages")
sys.path.append(r"/mnt/home/tudomlumleart/.local/lib/python3.10/site-packages/")
sys.path.append(r"/mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jupyter-gpu/lib/python3.10/site-packages")

import matplotlib.pyplot as plt
from functions import *
from utils import *
from sklearn.decomposition import PCA

import torch
import json
import multiprocessing
from cmdstanpy import CmdStanModel


def interpolate_polymers(polys):
    num_probes, num_coords, num_cells = polys.shape
    new_polys = np.zeros((num_probes, num_coords, num_cells))
    for c in range(num_cells):
        curr_cells = polys[:, :, c]
        for x in range(num_coords):
            curr_coords = curr_cells[:, x]
            missing_indices = np.isnan(curr_coords)
            valid_indices = ~missing_indices
            interp_coords = np.interp(np.flatnonzero(missing_indices), np.flatnonzero(valid_indices), curr_coords[valid_indices])
            new_polys[missing_indices, x, c] = interp_coords
            new_polys[valid_indices, x, c] = curr_coords[valid_indices]
    return new_polys


def calculate_variance_sample(dmap_list, dmap_ref, num_probes):
    variance_list = [num_probes**-2 * np.linalg.norm(x - dmap_ref)**2 for x in dmap_list]
    return np.mean(variance_list)


def run_mcmc(save_dir):
    # Load observations 
    # Load dataset from file
    print(f"Loading data ...")
    folder_path = '/mnt/home/tudomlumleart/ceph/05_Sox9Dataset/'
    
    num_monomers = 80
    
    # Load polys data and then perform linear interpolation
    # List all .mat files in the folder and load them
    cnc_polys = scipy.io.loadmat(folder_path + 'cncPols.mat')['cncPols'][:num_monomers, :, :]
    esc_polys = scipy.io.loadmat(folder_path + 'escPols.mat')['escPols'][:num_monomers, :, :]
    
    esc_polys_interp = interpolate_polymers(esc_polys)
    cnc_polys_interp = interpolate_polymers(cnc_polys)
    
    esc_maps_interp = np.array([squareform(pdist(esc_polys_interp[:80, :, i])) for i in range(esc_polys_interp.shape[2])])
    cnc_maps_interp = np.array([squareform(pdist(cnc_polys_interp[:80, :, i])) for i in range(cnc_polys_interp.shape[2])])
    esc_maps_interp_flat = np.array([x.flatten() for x in esc_maps_interp])
    cnc_maps_interp_flat = np.array([x.flatten() for x in cnc_maps_interp])
    all_maps_interp = np.concatenate((esc_maps_interp, cnc_maps_interp), axis=0)
    all_maps_interp_flat = np.concatenate((esc_maps_interp_flat, cnc_maps_interp_flat), axis=0)
    
    num_metastructures = 50
    
    print(f"Running PCA...")
    pca = PCA()
    pca.fit(all_maps_interp_flat)
    n_components = pca.n_components_
    esc_maps_pca = pca.transform(esc_maps_interp_flat)
    cnc_maps_pca = pca.transform(cnc_maps_interp_flat)
    
    for i in tqdm(range(1, n_components+1)):
        esc_df = pd.DataFrame(esc_maps_pca[:, i-1], columns=['PC{}'.format(i)])
        esc_df['label'] = 'ESC'
        cnc_df = pd.DataFrame(cnc_maps_pca[:, i-1], columns=['PC{}'.format(i)])
        cnc_df['label'] = 'CNC'
        all_df = pd.concat([esc_df, cnc_df], axis=0)
        
        lower_quantile = 0.01
        upper_quantile = 1 - lower_quantile
        pc_min = all_df['PC{}'.format(i)].quantile(lower_quantile)
        pc_max = all_df['PC{}'.format(i)].quantile(upper_quantile)
        
        pc_grid = np.linspace(pc_min, pc_max, num_metastructures)
        
        metastructure = np.array([x * pca.components_[i-1, :].reshape(1, -1) + pca.mean_ for x in pc_grid])
        templates_flatten = metastructure
        
        print(f"Calculating variance...")
        measurement_error_esc = [calculate_variance_sample(esc_maps_interp, x.reshape(80, 80), 80)**0.5 for x in templates_flatten]
        measurement_error_cnc = [calculate_variance_sample(cnc_maps_interp, x.reshape(80, 80), 80)**0.5 for x in templates_flatten]
        
        # Generate log prior for metastructures 
        print(f"Calculating prior...")
        lpm = [(logprior(x, num_monomers)).tolist() for x in templates_flatten]
        
        # Generate log likelihood for observations given metastructures 
        print(f"Calculating loglikelihood...")
        ll_esc = [[(loglikelihood(y, x, z, num_monomers)).tolist() for x, z in zip(templates_flatten, measurement_error_esc)] for y in esc_maps_interp_flat]
        ll_cnc = [[(loglikelihood(y, x, z, num_monomers)).tolist() for x, z in zip(templates_flatten, measurement_error_cnc)] for y in cnc_maps_interp_flat]
        
        N_cnc = cnc_maps_interp_flat.shape[0]
        N_esc = esc_maps_interp_flat.shape[0]
        M = templates_flatten.shape[0]
        
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
        
        conditions = ['ESC', 'CNC']
        for condition in conditions:
            output_dir = os.path.join(save_dir, condition, f'PC{i}')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Write json files for reading into stan program
            json_filename = os.path.join(output_dir, 'data.json')
            stan_output_file = os.path.join(output_dir, 'stan_output')
            
            if condition == 'ESC':
                data_dict = {
                    "M": M,
                    "N": N_esc,
                    "ll_map": ll_esc,
                    "lpm_vec": lpm,
                }
            elif condition == 'CNC': 
                data_dict = {
                    "M": M,
                    "N": N_cnc,
                    "ll_map": ll_cnc,
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
        
    
# Run file from the command line 
if __name__ == '__main__':
    save_dir = '/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/MCMC_results/20240904_WeightMCMC_AllPCs_Marginals_001Quantile/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
            
    print(f"Running MCMC ...")
    run_mcmc(save_dir)
    
    