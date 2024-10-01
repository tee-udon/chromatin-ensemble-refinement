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


def calculate_variance_sample(dmap_list, dmap_ref):
    variance_list = [np.linalg.norm(x - dmap_ref) for x in dmap_list]
    return np.mean(variance_list)


def run_mcmc(save_dir, var):
    # Load observations 
    # Load dataset from file
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
    
    pca = PCA(n_components=2)
    pca.fit(all_maps_interp_flat)
    esc_maps_pca = pca.transform(esc_maps_interp_flat)
    cnc_maps_pca = pca.transform(cnc_maps_interp_flat)
    
    esc_df = pd.DataFrame(esc_maps_pca, columns=['PC1', 'PC2'])
    esc_df['label'] = 'ESC'
    cnc_df = pd.DataFrame(cnc_maps_pca, columns=['PC1', 'PC2'])
    cnc_df['label'] = 'CNC'
    all_df = pd.concat([esc_df, cnc_df], axis=0)
    
    # Find 0.05 and 0.95 quantiles of PC1 and PC2 data
    pc1_05 = all_df['PC1'].quantile(0.01)
    pc1_95 = all_df['PC1'].quantile(0.99)
    pc2_05 = all_df['PC2'].quantile(0.01)
    pc2_95 = all_df['PC2'].quantile(0.99)
    
    pc1_grid = np.linspace(pc1_05, pc1_95, 50)
    pc2_grid = np.linspace(pc2_05, pc2_95, 50)
    
    # Generate combination of pc1 and pc2 values
    pc1_grid, pc2_grid = np.meshgrid(pc1_grid, pc2_grid)
    
    # put this into a dataframe
    pc1_grid_flat = pc1_grid.flatten()
    pc2_grid_flat = pc2_grid.flatten()
    pc1_pc2_df = pd.DataFrame({'PC1': pc1_grid_flat, 'PC2': pc2_grid_flat})
    pc1_pc2_df['label'] = 'metastructures'
    
    # Sort PC2 in descending order while keeping PC1 in ascending order
    pc1_pc2_df = pc1_pc2_df.sort_values(by=['PC1', 'PC2'], ascending=[True, False], ignore_index=True)  

    metastr_from_pca = pca.inverse_transform(pc1_pc2_df[['PC1', 'PC2']])   

    templates_flatten = metastr_from_pca
    
    measurement_error_esc = [var**0.5 for x in templates_flatten]
    measurement_error_cnc = [var**0.5 for x in templates_flatten]
    measurement_error_all = [var**0.5 for x in templates_flatten]
    
    # Generate log prior for metastructures 
    lpm = [(logprior(x, num_monomers)).tolist() for x in templates_flatten]
    
    # Generate log likelihood for observations given metastructures 
    ll_esc = [[(loglikelihood(y, x, z, num_monomers)).tolist() for x, z in zip(templates_flatten, measurement_error_esc)] for y in esc_maps_interp_flat]
    ll_cnc = [[(loglikelihood(y, x, z, num_monomers)).tolist() for x, z in zip(templates_flatten, measurement_error_cnc)] for y in cnc_maps_interp_flat]
    ll_all = [[(loglikelihood(y, x, z, num_monomers)).tolist() for x, z in zip(templates_flatten, measurement_error_all)] for y in all_maps_interp_flat]
    
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
    
    conditions = ['ESC', 'CNC', 'all']
    for condition in conditions:
        output_dir = os.path.join(save_dir, condition)
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
        elif condition == 'all':
            data_dict = {
                "M": M,
                "N": N_esc + N_cnc,
                "ll_map": ll_all,
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
    save_dir = '/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/MCMC_results/20240809_WeightMCMC_PCA_metastructures_var/'
        
    var_list = np.logspace(6, 10, 5)
            
    print(f"Running MCMC ...")
    for v in var_list:
        print('Current variance: ', v)
        if not os.path.exists(save_dir + f'var_{v}/'):
            os.makedirs(save_dir + f'var_{v}/')
        run_mcmc(save_dir + f'var_{v}/', v)
    
     