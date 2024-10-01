# Load necessary modules and dataset 
import sys
sys.path.append(r"/mnt/ceph/users/tudomlumleart/00_VirtualEnvironments/teeu/lib/python3.10/site-packages")
sys.path.append(r"/mnt/home/tudomlumleart/.local/lib/python3.10/site-packages/")
sys.path.append(r"/mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jupyter-gpu/lib/python3.10/site-packages")

import matplotlib.pyplot as plt
from functions import *
from utils import *

from joblib import Parallel, delayed
import pylops
import pyproximal

import csv


def matrix_completion_k(k):
    bc1_nan_map = observations_flatten_with_nan[:, num_probes*k:num_probes*(k+1)]
    bc1_map = observations_flatten[:, num_probes*k:num_probes*(k+1)]
    
    # This is the indices of existing data
    existing_indices = np.where(existing_data_flatten[:, num_probes*k:num_probes*(k+1)].ravel())
    
    nx, ny = bc1_nan_map.shape
    Rop = pylops.Restriction(ny*nx, 
                         existing_indices[0], dtype='float64')
    
    Uy, Sy, Vhy = np.linalg.svd(bc1_nan_map, full_matrices=False)
    
    mu1 = 0.5 * np.sum(Sy)
    y = Rop * bc1_map.ravel()
    f = pyproximal.L2(Rop, y)
    g = pyproximal.proximal.NuclearBall((ny, nx), mu1)

    Xpgc = pyproximal.optimization.primal.ProximalGradient(f, g, np.zeros(ny*nx), acceleration='vandenberghe',
                                                        tau=1., niter=100, show=False)
    Xpgc = Xpgc.reshape(nx, ny)
    
    return Xpgc


def matrix_completion_surface_k(k):
    bc1_nan_map = observations_flatten_with_nan[:, num_probes*k:num_probes*(k+1)]
    bc1_map = observations_flatten[:, num_probes*k:num_probes*(k+1)]
    
    # This is the indices of existing data
    existing_indices = np.where(existing_data_flatten[:, num_probes*k:num_probes*(k+1)].ravel())
    
    nx, ny = bc1_nan_map.shape
    Rop = pylops.Restriction(ny*nx, 
                         existing_indices[0], dtype='float64')

    Uy, Sy, Vhy = np.linalg.svd(bc1_nan_map, full_matrices=False)
    
    mu = 1
    y = Rop * bc1_map.ravel()
    f = pyproximal.L2(Rop, y)
    g = pyproximal.Nuclear((ny, nx), mu)

    Xpg = pyproximal.optimization.primal.ProximalGradient(f, g, np.zeros(ny*nx), acceleration='vandenberghe',
                                                        tau=1., niter=100, show=False)
    Xpg = Xpg.reshape(nx, ny)

    return Xpg


def interpolate_polymers(polys):
    # Notice here that the input polymer shape is different!!
    num_cells, num_probes, num_coords = polys.shape
    new_polys = np.zeros((num_probes, num_coords, num_cells))
    for c in range(num_cells):
        curr_cells = polys[c, :, :]
        for x in range(num_coords):
            curr_coords = curr_cells[:, x]
            missing_bool = np.isnan(curr_coords)
            valid_indices = np.where(~missing_bool)[0]
            missing_indices = np.where(missing_bool)[0]
            # print(missing_indices, valid_indices, curr_coords[valid_indices])
            interp_coords = np.interp(missing_indices, valid_indices, curr_coords[valid_indices])
            new_polys[missing_indices, x, c] = interp_coords
            new_polys[valid_indices, x, c] = curr_coords[valid_indices]
    return new_polys


    
    
if __name__ == '__main__':
    # Accept the path to a pickle file as an argument
    p = sys.argv[1]
    
    # Load the dataset from that pickle file 
    print("Loading dataset...")
    print(p)
    dataset_list, param_dict = load_dataset(p)
    
    # Extract file name from the path 
    core_name = p.split("/")[-1].split(".")[0]
    save_folder = '/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/MatrixCompletion_results/20240724' 
    save_dir = os.path.join(save_folder, f'{core_name}.txt')
    print(f"Saving results to {save_dir}")
    
    print('Initializing dataset...')
    mc_pred_Sy_err_list = []
    mc_pred_surface_err_list = []
    mean_impute_err_list = []
    
    for i in tqdm(range(50)):
        print(f"Running Matrix Completion for dataset {i}...")
        template_list = dataset_list[i]['template_chain_list']
        X = dataset_list[i]['observation_list']
        label_list = dataset_list[i]['labels']

        observations_flatten = [squareform(pdist(x)).flatten() for x in X]

        # generate weight of each label from label_list
        true_weights = np.array([np.sum(label_list == i) for i in np.unique(label_list)]) / len(label_list)
        templates_flatten = [squareform(pdist(x)).flatten() for x in template_list]

        # Generate random walk for guessing structures
        num_monomers = param_dict['num_monomers']
        mean_bond_length = param_dict['mean_bond_length']
        std_bond_length = param_dict['std_bond_length'] 
        num_templates = param_dict['num_templates']
        measurement_error = param_dict['noise_std']
        num_observations = param_dict['num_observations']
        num_probes = num_monomers
        num_candidates = num_templates
        
        # Introducing missing data to the dataset at specific barcodes with some probabiliity between [0, 1]
        missing_data_prob = np.random.normal(0.2, 0.3, num_monomers)
        missing_data_prob[missing_data_prob < 0] = 0
        missing_data_prob[missing_data_prob > 1] = 1
        plt.plot(missing_data_prob)
        
        # Get shuffle indices for shuffling dataset 
        shuffle_indices = np.arange(num_observations)
        np.random.shuffle(shuffle_indices)

        observations_with_nan = np.array([squareform(pdist(x)) for x in X])
        observations_with_nan = observations_with_nan[shuffle_indices, :, :]
        polys_with_nan = np.array([x for x in X])
        polys_with_nan = polys_with_nan[shuffle_indices, :, :]
        existing_data = np.zeros(observations_with_nan.shape)
        # Randomly add nan to the dataset based on the missing_data_prob
        for i in range(num_observations):
            missing_indices = np.random.uniform(size=num_monomers) < missing_data_prob
            observations_with_nan[i, missing_indices, :] = 0
            observations_with_nan[i, :, missing_indices] = 0
            polys_with_nan[i, missing_indices, :] = np.nan
            existing_data[i, ~missing_indices, :] = 1
            existing_data[i, :, ~missing_indices] = 1
            
        existing_data_flatten = np.array([x.flatten() for x in existing_data.astype(bool)])
        observations_flatten = np.array([squareform(pdist(x)) for x in X])[shuffle_indices, :, :]
        observations_flatten = np.array([x.flatten() for x in observations_flatten])
        observations_flatten_with_nan = np.array([x.flatten() for x in observations_with_nan])
        mean_impute_polys = interpolate_polymers(polys_with_nan)
        mean_impute_maps = np.array([squareform(pdist(mean_impute_polys[:, :, x])) for x in range(mean_impute_polys.shape[2])])
        mean_impute_flatten = np.array([x.flatten() for x in mean_impute_maps])
        
        print('Running matrix completion Nuclear L1 Ball...')
        mc_pred_Sy = Parallel(n_jobs=-1)(delayed(matrix_completion_k)(k) for k in range(num_probes))
        print('Running matrix completion Nuclear Surface...')
        mc_pred_surface = Parallel(n_jobs=-1)(delayed(matrix_completion_surface_k)(k) for k in range(num_probes))
        
        mc_pred_Sy = np.hstack(mc_pred_Sy)
        mc_pred_surface = np.hstack(mc_pred_surface)
        
        print('Calculating performance metrics...')
        mc_pred_Sy_err = np.linalg.norm(mc_pred_Sy - observations_flatten)
        mc_pred_surface_err = np.linalg.norm(mc_pred_surface - observations_flatten)
        mean_impute_err = np.linalg.norm(mean_impute_flatten - observations_flatten)
        
        # append errors to lists
        mc_pred_Sy_err_list.append(mc_pred_Sy_err)
        mc_pred_surface_err_list.append(mc_pred_surface_err)
        mean_impute_err_list.append(mean_impute_err)
        
        
    print('Write performance matrix to the a csv file...')
    
    # Write the list of errors in a csv file
    # Each column takes each list of errors
    # The first row includes the name of the list
    rows = zip(mc_pred_Sy_err_list, mc_pred_surface_err_list, mean_impute_err_list)
    with open(f'{save_dir}', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['nuclear_l1_ball', 'nuclear_surface', 'linear_imputation'])  # Write header
        writer.writerows(rows)  # Write the rows
            
    print('Done')