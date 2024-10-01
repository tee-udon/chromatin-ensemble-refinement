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

# Use Low Rank completion from pylop and pyproximal 
import pylops
import pyproximal


def mc_ball_Sx(k):
    bc1_nan_map = observations_flatten_with_nan[:, num_probes*k:num_probes*(k+1)]
    bc1_map = observations_flatten[:, num_probes*k:num_probes*(k+1)]
    
    # This is the indices of existing data
    existing_indices = np.where(existing_data_flatten[:, num_probes*k:num_probes*(k+1)].ravel())
    
    nx, ny = bc1_map.shape
    Rop = pylops.Restriction(ny*nx, 
                         existing_indices[0], dtype='float64')
    
    Ux, Sx, Vhx = np.linalg.svd(bc1_map, full_matrices=False)
    Uy, Sy, Vhy = np.linalg.svd(bc1_nan_map, full_matrices=False)
    
    mu1 = 0.8 * np.sum(Sx)
    y = Rop * bc1_map.ravel()
    f = pyproximal.L2(Rop, y)
    g = pyproximal.proximal.NuclearBall((ny, nx), mu1)

    Xpgc = pyproximal.optimization.primal.ProximalGradient(f, g, np.zeros(ny*nx), acceleration='vandenberghe',
                                                        tau=1., niter=100, show=False)
    Xpgc = Xpgc.reshape(nx, ny)

    return Xpgc


def mc_ball_Sy(k):
    bc1_nan_map = observations_flatten_with_nan[:, num_probes*k:num_probes*(k+1)]
    bc1_map = observations_flatten[:, num_probes*k:num_probes*(k+1)]
    
    # This is the indices of existing data
    existing_indices = np.where(existing_data_flatten[:, num_probes*k:num_probes*(k+1)].ravel())
    
    nx, ny = bc1_map.shape
    Rop = pylops.Restriction(ny*nx, 
                         existing_indices[0], dtype='float64')
    
    Ux, Sx, Vhx = np.linalg.svd(bc1_map, full_matrices=False)
    Uy, Sy, Vhy = np.linalg.svd(bc1_nan_map, full_matrices=False)
    
    mu1 = 0.8 * np.sum(Sy)
    y = Rop * bc1_map.ravel()
    f = pyproximal.L2(Rop, y)
    g = pyproximal.proximal.NuclearBall((ny, nx), mu1)

    Xpgc = pyproximal.optimization.primal.ProximalGradient(f, g, np.zeros(ny*nx), acceleration='vandenberghe',
                                                        tau=1., niter=100, show=False)
    Xpgc = Xpgc.reshape(nx, ny)

    return Xpgc


def mc_surface(k):
    bc1_nan_map = observations_flatten_with_nan[:, num_probes*k:num_probes*(k+1)]
    bc1_map = observations_flatten[:, num_probes*k:num_probes*(k+1)]
    
    # This is the indices of existing data
    existing_indices = np.where(existing_data_flatten[:, num_probes*k:num_probes*(k+1)].ravel())
    
    nx, ny = bc1_map.shape
    Rop = pylops.Restriction(ny*nx, 
                         existing_indices[0], dtype='float64')
    
    Ux, Sx, Vhx = np.linalg.svd(bc1_map, full_matrices=False)
    Uy, Sy, Vhy = np.linalg.svd(bc1_nan_map, full_matrices=False)
    
    mu = 1
    y = Rop * bc1_map.ravel()
    f = pyproximal.L2(Rop, y)
    g = pyproximal.Nuclear((ny, nx), mu)

    Xpg = pyproximal.optimization.primal.ProximalGradient(f, g, np.zeros(ny*nx), acceleration='vandenberghe',
                                                        tau=1., niter=100, show=False)
    Xpg = Xpg.reshape(nx, ny)
    
    return Xpg

def mean_inputation(nan_2d_array):
    result = nan_2d_array.copy()
    for i in range(nan_2d_array.shape[0]):
        nan_indices = []
        for z, x in enumerate(nan_2d_array[i, :]):
            if x == 0 and z > 0:
                nan_indices.append(z)
            else:
                if not nan_indices:
                    prev_exist = x
                else:
                    diff = (x - prev_exist) / (len(nan_indices)+1)
                    for k, j in enumerate(nan_indices):
                        result[i, j] = prev_exist + diff*(k+1)
                    prev_exist = x
                    nan_indices = []
    return result

# Run file from the command line 
if __name__ == '__main__':
    # Accept the path to a pickle file as an argument
    p = '/mnt/home/tudomlumleart/ceph/03_GaussianChainSimulation/20240627/dataset_100_10_20_10_1000_40.0_10000.pkl'
    
    # Load the dataset from that pickle file 
    print("Loading dataset...")
    dataset_list, param_dict = load_dataset(p)
    
    # Generate save directory for the result
    save_dir = '/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/MatrixCompletion_results/'
    
    template_list = dataset_list[0]['template_chain_list']
    X = dataset_list[0]['observation_list']
    label_list = dataset_list[0]['labels']

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
        
    missing_data_prob = np.random.normal(0.1, 0.5, num_monomers)
    missing_data_prob[missing_data_prob < 0] = 0
    missing_data_prob[missing_data_prob > 1] = 1
    
    # Get shuffle indices for shuffling dataset 
    shuffle_indices = np.arange(num_observations)
    np.random.shuffle(shuffle_indices)
    
    observations_with_nan = np.array([squareform(pdist(x)) for x in X])
    observations_with_nan = observations_with_nan[shuffle_indices, :, :]
    existing_data = np.zeros(observations_with_nan.shape)
    # Randomly add nan to the dataset based on the missing_data_prob
    for i in range(num_observations):
        missing_indices = np.random.uniform(size=num_monomers) < missing_data_prob
        observations_with_nan[i, missing_indices, :] = 0
        observations_with_nan[i, :, missing_indices] = 0
        existing_data[i, ~missing_indices, :] = 1
        existing_data[i, :, ~missing_indices] = 1
        
    existing_data_flatten = np.array([x.flatten() for x in existing_data.astype(bool)])
    observations_flatten = np.array([squareform(pdist(x)) for x in X])[shuffle_indices, :, :]
    observations_flatten = np.array([x.flatten() for x in observations_flatten])
    observations_flatten_with_nan = np.array([x.flatten() for x in observations_with_nan])
    
    n_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {n_cores}")
    
    pool = multiprocessing.Pool()
    mc_Sx_mp = pool.map(mc_ball_Sx, range(num_probes))
    mc_Sy_mp = pool.map(mc_ball_Sy, range(num_probes))
    mc_surface_mp = pool.map(mc_surface, range(num_probes))
    
    def mean_imputation_k(k):
        bc1_nan_map = observations_flatten_with_nan[:, num_probes*k:num_probes*(k+1)]
        return mean_inputation(bc1_nan_map)
    
    mi_pred = [mean_imputation_k(x) for x in tqdm(range(num_probes))]
    
    mc_Sx_err = np.linalg.norm(np.hstack(mc_Sx_mp) - observations_flatten)
    mc_Sy_err = np.linalg.norm(np.hstack(mc_Sy_mp) - observations_flatten)
    mc_surface_err = np.linalg.norm(np.hstack(mc_surface_mp) - observations_flatten)
    mi_err = np.linalg.norm(np.hstack(mi_pred) - observations_flatten)
    
    # Write missing data probability to the text file and write the error to the text file
    with open(save_dir + 'missing_data_prob.txt', 'w') as f:
        f.write(str(missing_data_prob) + '\n')
        # Write the error to the text file
        f.write(f"mc_Sx_err: {mc_Sx_err}\n")
        f.write(f"mc_Sy_err: {mc_Sy_err}\n")
        f.write(f"mc_surface_err: {mc_surface_err}\n")
        f.write(f"mi_err: {mi_err}\n")
        