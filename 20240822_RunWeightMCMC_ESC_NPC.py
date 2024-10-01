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


def calculate_variance_sample(dmap_list, dmap_ref, num_probes):
    variance_list = [num_probes**-2 * np.linalg.norm(x - dmap_ref)**2 for x in dmap_list]
    return np.mean(variance_list)



def generate_distance_maps(polys):
    num_probes, num_coords, num_cells = polys.shape
    return np.array([squareform(pdist(polys[:, :, c])) for c in range(num_cells)])


def run_mcmc(save_dir):
    # Load observations 
    # Load dataset from file
    folder_path = '/mnt/home/tudomlumleart/ceph/08_SedonaDataset'
    dataset_path = os.path.join(folder_path, 'fig5_data_brain_esc_npc.mat')
    
    num_monomers = 54
    
    dataset = scipy.io.loadmat(dataset_path)
    
    brain_poly1 = dataset['brainPoly1']
    brain_poly2 = dataset['brainPoly2']
    esc_poly1 = dataset['esc1_polys']
    esc_poly2 = dataset['esc2_polys']
    npc_poly1 = dataset['npc1_polys']
    npc_poly2 = dataset['npc2_polys']
    
    brain_poly1 = interpolate_polymers(brain_poly1)
    brain_poly2 = interpolate_polymers(brain_poly2)
    esc_poly1 = interpolate_polymers(esc_poly1)
    esc_poly2 = interpolate_polymers(esc_poly2)
    npc_poly1 = interpolate_polymers(npc_poly1)
    npc_poly2 = interpolate_polymers(npc_poly2)
    
    brain_map1 = generate_distance_maps(brain_poly1)
    brain_map2 = generate_distance_maps(brain_poly2)
    esc_map1 = generate_distance_maps(esc_poly1)
    esc_map2 = generate_distance_maps(esc_poly2)
    npc_map1 = generate_distance_maps(npc_poly1)
    npc_map2 = generate_distance_maps(npc_poly2)
    
    brain_map = np.concatenate([brain_map1, brain_map2], axis=0)
    esc_map = np.concatenate([esc_map1, esc_map2], axis=0)
    npc_map = np.concatenate([npc_map1, npc_map2], axis=0)
    
    esc_map_flat = np.array([x.flatten() for x in esc_map])
    npc_map_flat = np.array([x.flatten() for x in npc_map]) 
    
    all_map_flat = np.concatenate([esc_map_flat, npc_map_flat], axis=0)
    
    pca = PCA(n_components=2)
    pca.fit(all_map_flat)
    esc_map_pca = pca.transform(esc_map_flat)
    npc_map_pca = pca.transform(npc_map_flat)    
    
    esc_df = pd.DataFrame(esc_map_pca, columns=['PC1', 'PC2'])
    esc_df['label'] = 'ESC'
    npc_df = pd.DataFrame(npc_map_pca, columns=['PC1', 'PC2'])
    npc_df['label'] = 'NPC'
    
    all_df = pd.concat([esc_df, npc_df], axis=0)
    
    # Find 0.05 and 0.95 quantiles of PC1 and PC2 data
    l = 0.01
    u = 1-0.01

    pc1_l = all_df['PC1'].quantile(l)
    pc1_u = all_df['PC1'].quantile(u)
    pc2_l = all_df['PC2'].quantile(l)
    pc2_u = all_df['PC2'].quantile(u)

    pc1_grid = np.linspace(pc1_l, pc1_u, 50)
    pc2_grid = np.linspace(pc2_l, pc2_u, 50)

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
    
    measurement_error_esc = [calculate_variance_sample(esc_map, x.reshape(num_monomers, num_monomers), num_monomers)**0.5 for x in templates_flatten]
    measurement_error_npc = [calculate_variance_sample(npc_map, x.reshape(num_monomers, num_monomers), num_monomers)**0.5 for x in templates_flatten]
    
    # Generate log prior for metastructures 
    lpm = [(logprior(x, num_monomers)).tolist() for x in templates_flatten]
    
    # Generate log likelihood for observations given metastructures 
    ll_esc = [[(loglikelihood(y, x, z, num_monomers)).tolist() for x, z in zip(templates_flatten, measurement_error_esc)] for y in esc_map_flat]
    ll_npc = [[(loglikelihood(y, x, z, num_monomers)).tolist() for x, z in zip(templates_flatten, measurement_error_npc)] for y in npc_map_flat]
    
    N_npc = npc_map_flat.shape[0]
    N_esc = esc_map_flat.shape[0]
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
    
    conditions = ['ESC', 'NPC']
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
        elif condition == 'NPC': 
            data_dict = {
                "M": M,
                "N": N_npc,
                "ll_map": ll_npc,
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
    save_dir = '/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/MCMC_results/20240822_WeightMCMC_ESC_NPC/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
            
    print(f"Running MCMC ...")
    run_mcmc(save_dir)
    
     