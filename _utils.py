import sys

sys.path.append(r"/mnt/ceph/users/tudomlumleart/00_VirtualEnvironments/teeu/lib/python3.10/site-packages")
sys.path.append(r"/mnt/home/tudomlumleart/.local/lib/python3.10/site-packages/")
sys.path.append(r"/mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jupyter-gpu/lib/python3.10/site-packages")

import os 
import h5py
import ast 
import math
import copy
import csv 
import scipy
import shutil
import multiprocessing
import umap
import json
import time 
import sklearn
import numpy 
import pickle
import itertools
import jax
import jax.numpy as jnp
import seaborn as sns
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from jax import lax
from functools import partial
from numpy.typing import ArrayLike
from typing import Literal
from tqdm.auto import tqdm
from scipy import spatial 
from matplotlib import cm 
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.interpolate import CubicSpline
from cmdstanpy import CmdStanModel
from sklearn.decomposition import PCA


def is_gpu_available():
    """
    Check if JAX detects a GPU on the system.

    Returns:
    bool: True if a GPU is available, False otherwise.
    """
    devices = jax.devices()
    for device in devices:
        if device.device_kind == "gpu":
            return True
    return False


def generate_gaussian_chain(num_monomers: int, 
                            mean_bond_length: float, 
                            std_bond_length: float) -> np.ndarray:
    """Generate a Gaussian chain polymer 
    
    Parameters
    ----------
    num_monomers : int 
                  number of polymers in a chain 
    mean_bond_length : float
                      average distance between two neighboring monomers 
    std_bond_length : float
                     standard deviation of distance between two neighboring monomers 
    
    Returns
    ------
    ArrayLike
        an `num_monomers-by-3` numpy.ndarray of monomer coordinates
        
    """ 
    # Generate steps: each step is a 3D vector 
    steps = np.random.normal(mean_bond_length, std_bond_length, size=(num_monomers, 3))

    # Compute positions by cumulative sum of steps
    positions = np.cumsum(steps, axis=0)
    
    return positions


def visualize_polymer(polymer_chain: ArrayLike, save_path: str = '') -> None:
    """Plot a polymer chain in 3D space and optionally save the figure 
    
    Monomers are colored by the distance from one end of the polymer. 
    
    Parameters
    ----------
    polymer_chain : numpy.ndarray
                    an `num_monomers-by-3` ArrayLike of monomer coordinates 
    save_path : str, optional
                path to save the figure
                
    """
    # Extract each coordinate of this polymer chain 
    x = polymer_chain[:, 0]
    y = polymer_chain[:, 1]
    z = polymer_chain[:, 2]
    
    # Intepolate path between monomer to show connectivity within the polymer
    # Parameterize by cumulative distance along the points
    t = np.zeros(x.shape)
    t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2 + (z[1:] - z[:-1])**2)
    t = np.cumsum(t)
    
    # Create a cubic spline interpolation for each dimension
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    cs_z = CubicSpline(t, z)
    
    # Normalize the monomer number 
    num_monomers = polymer_chain.shape[0] 
    norm_monomer_number = np.arange(0, num_monomers) / num_monomers
    monomer_colors = cm.rainbow(norm_monomer_number)

    # Generate fine samples for a smooth curve
    t_fine = np.linspace(t[0], t[-1], 500)
    x_fine = cs_x(t_fine)
    y_fine = cs_y(t_fine)
    z_fine = cs_z(t_fine)

    # Create a new matplotlib figure and an axes instance (the 3d part)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot using scatter
    ax.plot(x_fine, y_fine, z_fine, 'gray', label='Interpolated Path')
    for i in range(num_monomers):
        ax.scatter(x[i], y[i], z[i], color=monomer_colors[i], s=50, alpha=0.75) 

    # Labeling the axes (optional but recommended for clarity)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # Create a ScalarMappable with the same colormap and normalization as the scatter
    # sm = cm.ScalarMappable(cmap='rainbow', norm=norm_monomer_number)
    # sm.set_array([]) 
    
    # Add colorbar
    # cbar = plt.colorbar(sm, ax=ax)
    # cbar.set_label('Monomer number')
    
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def compare_distance_maps(
        chain1: ArrayLike, chain2: ArrayLike, 
        type1: Literal['polymer', 'distance_map', 'flatten_distance_map'] = 'polymer', 
        type2: Literal['polymer', 'distance_map', 'flatten_distance_map'] = 'polymer',
        save_path: str = ''):
    """ Plot distance maps of `chain1` and `chain2` side-by-side and optionally save the figure.
    
    The shapes of `chain1` and `chain2` depend on `type1` and `type2`. 
    
    Parameters
    ----------
    chain1 : ArrayLike
             an array-like object represents the first polymer. Its shape depends on `type1`.
             
             `(num_monomers, 3)` if `polymer`
             `(num_monomers, num_monomers)` if `distance_map`
             `(num_monomers**2, 1)` if 'flatten_distance_map`
             
    
    chain2 : ArrayLike
             an array-like object represents the second polymer. Its shape depends on `type2`.
             
             `(num_monomers, 3)` if `polymer`
             `(num_monomers, num_monomers)` if `distance_map`
             `(num_monomers**2, 1)` if 'flatten_distance_map`
    
    type1 : {`polymer`, `distance_map`, `flatten_distance_map`}, default=`polymer`
            a string represents the input shape for `chain1`
            
    type2 : {`polymer`, `distance_map`, `flatten_distance_map`}, default=`polymer`
            a string represents the input shape for `chain2`
            
    save_path : str, optional
                path to save the figure

    """
    assert type1 in ('polymer', 'distance_map', 'flatten_distance_map')
    assert type2 in ('polymer', 'distance_map', 'flatten_distance_map')
    
    # Calculate pairwise distance maps
    if type1 == 'polymer':
        distance_map1 = squareform(pdist(chain1))
    elif type1 == 'flatten_distance_map':
        distance_map1 = np.reshape(chain1, [round(np.sqrt(chain1.shape[0])), round(np.sqrt(chain1.shape[0]))])
    else:
        distance_map1 = chain1
        
    if type2 == 'polymer':
        distance_map2 = squareform(pdist(chain2)) 
    elif type2 == 'flatten_distance_map':
        distance_map2 = np.reshape(chain2, [round(np.sqrt(chain2.shape[0])), round(np.sqrt(chain2.shape[0]))])
    else:
        distance_map2 = chain2
    
    # Initialize new figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot distance maps 
    heatmap1 = ax1.imshow(distance_map1, cmap='hot', aspect='auto')
    ax1.set_title('Chain 1')
    cb1 = fig.colorbar(heatmap1, ax=ax1)
    cb1.set_label('Euclidean distance [a.u.]')
    
    heatmap2 = ax2.imshow(distance_map2, cmap='hot', aspect='auto')
    ax2.set_title('Chain 2')
    cb2 = fig.colorbar(heatmap2, ax=ax2)
    cb2.set_label('Euclidean distance [a.u.]')
    
    # Display the figure
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_observations(polymer_chain: ArrayLike, 
                          num_observations: int, 
                          gaussian_noise_std: float) -> list:
    """ Given a template `polymer_chain`, generate `num_observations` polymer chains by adding 
    some gaussian noise with zero mean and `gaussian_noise_std` standard deviation to the 
    `polymer_chain`
    
    Parameters
    ----------
    polymer_chain : ArrayLike
                   an `(num_monomers-by-3)` array containing the monomer coordinates
    num_observations : int
                       a number of independent observations generated by this function 
    gaussian_noise_std : float
                         the standard deviation for the gaussian noise to be used to generate observations
    
    Returns
    ------
    list of numpy.ndarray 
        a list of `num_obervations` numpy.ndarray with the same shape as `polymer_chain` 
    """
    observation_list = []
    polymer_size = polymer_chain.shape 
    
    # Parameters for Gaussian noise 
    mean = 0 
    std = gaussian_noise_std
    
    for i in range(num_observations):
        # Generate Gaussian Noise 
        noise = np.random.normal(mean, std, polymer_size)
        
        # Add noise to the original data
        noisy_data = polymer_chain + noise 
        
        # Append this observation to the list
        observation_list.append(noisy_data)
    
    return observation_list


def visualize_dmap(dmap: ArrayLike, 
                   vmax: float = None, 
                   type_: Literal['polymer', 'distance_map', 'flatten_distance_map'] = 'distance_map',
                   save_path: str = '') -> None:
    """Plot a distance map, the upper bound of the colormap can be optionally set, and can save figure given a path.
    
    Parameters
    ----------
    dmap : ArrayLike
           an `num_monomers-by-num_monomers` pairwise euclidean distance array
    vmax : float, optional
           an upper bound for the colormap 
    type_ : {`polymer`, `distance_map`, `flatten_distance_map`}, default=`distance_map`
           a string represents the input shape for `dmap`
    save_path: str, optional 
               path to save the figure
    """
    # Create a new matplotlib figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if type_ == 'polymer':
        dmap = squareform(pdist(dmap))
    elif type_ == 'flatten_distance_map':
        dmap = np.reshape(dmap, [round(np.sqrt(dmap.shape[0])), round(np.sqrt(dmap.shape[0]))])
    
    # Plot using imshow
    if vmax is None: 
        heatmap = ax.imshow(dmap, cmap='hot', aspect='auto', vmin=0)
    else:
        heatmap = ax.imshow(dmap, cmap='hot', aspect='auto', vmin=0, vmax=vmax)
        
    cb = fig.colorbar(heatmap, ax=ax)
    cb.set_label('Euclidean distance [a.u.]')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    
# Helper function for generate_dataset function
def generate_dataset_parallel(combo):
    num_monomers, mean_bond_length, std_bond_length, num_templates, weight_dist, noise_std, num_observations, num_dataset, _ = combo[0]
    save_dir = combo[1]
    
    dataset_list = []
    for idx_dataset in range(num_dataset):
        # Generate template chains 
        template_chain_list = [generate_gaussian_chain(num_monomers, mean_bond_length, std_bond_length) for i in range(num_templates)]
        
        # Generate observations based on weight distribution, num_observations, and noise std
        # Calculate the concentration parameter alpha based on weight_dist
        # When weight_dist is 0, alpha should be very small (resulting in more uneven numbers)
        # When weight_dist is 1, alpha should be large (resulting in more even numbers)
        alpha = np.maximum(np.ones(num_templates) * weight_dist * 100, np.ones(num_templates)) 
        num_observation_list = np.int32(np.random.dirichlet(alpha) * num_observations) + 1
        observation_list = [generate_observations(c, n, noise_std) for c, n in zip(template_chain_list, num_observation_list)]
        
        # Concatenate observation_list into a single array
        observation_list = np.concatenate([*observation_list])
        labels_true = np.concatenate([[i for _ in range(j)] for i, j in enumerate(num_observation_list)])
        
        dataset_list.append({
                'template_chain_list': template_chain_list,
                'observation_list': observation_list,
                'labels': labels_true
            })
        
    # Create a pickle file to save the dataset
    pickle_file = save_dir + f"dataset_{num_monomers}_{mean_bond_length}_{std_bond_length}_{num_templates}_{weight_dist}_{noise_std}_{num_observations}.pkl"
    tmp_file = pickle_file + '.tmp'
    
    with open(tmp_file, 'wb') as f:
        pickle.dump(dataset_list, f)
    os.rename(tmp_file, pickle_file)

# Define a function that generate a dataset of polymer chains 
# Save structures and labels in a pickle file 
# Vary the following parameters 
# Number of groups, weights distribution, Gaussian_noise_std 
def generate_dataset(params_dict: dict) -> None:
    """
    Generate a dataset of polymer chains with different parameters. 
    
    The dataset will be saved at `save_dir` as pickle files with README.txt file containing the parameters used to generate the dataset.
    

    Parameters
    ----------
    params_dict : dict 
        A dictionary where keys are parameter names and values are lists of parameter values.
        The expected keys and their corresponding values are:
        
        - 'num_monomers' : a list of single int
            Number of monomers in each polymer chain. Example: 100
        - 'mean_bond_length' : a list of a single float
            Mean bond length of the polymer chain. Example: 1.0
        - 'std_bond_length' : a list of a single float
            Standard deviation of the bond length of the polymer chain. Example: 0.1
        - 'num_templates' : ArrayLike of int
            Number of template polymers used to generate the dataset. Example: [10, 20, 30]
        - 'weights_dist' : ArrayLike of float between 0 and 1
            Degree of evenness for weight distribution (0 for uneven, inf for even). Example: [0.1, 0.2, 0.3]
        - 'noise_std' : ArrayLike of float
            Standard deviation of the Gaussian noise used to generate observations from each template. Example: [0.1, 0.2, 0.3]
        - 'num_observations' : a list of a single int
            Total number of polymers generated in the dataset. Example: 1000
        - 'num_datasets' : a list of a single int
            Number of datasets to generate. Example: 1 
        - 'save_dir': a list of a single str
            Directory to save the dataset. Example: 'data/'

            
    Notes:
    ------
    This function generates a combinatorial dataset of polymers based on the provided parameter values.
    Ensure that the values for each parameter are provided as lists.
    """
    # Check number of CPU cores
    n_cores = os.cpu_count()
    if n_cores > 1:
        print(f"Using {n_cores} cores to generate the dataset.")
    else:
        print("Using 1 core to generate the dataset.")
    
    
    # Extract names and values of parameters 
    param_names = list(params_dict.keys())
    param_values = list(params_dict.values()) 
    
    # Check if all the parameters are provided
    assert len(param_names) == 9, "Please provide all the parameters"
    
    # Generate all possible combinations of parameter values without save_dir
    param_combinations = list(itertools.product(*param_values))
    
    # Create the save directory if it does not exist
    save_dir = params_dict['save_dir'][0]
    # Make sure the save_dir ends with a '/'
    if not save_dir.endswith('/'):
        save_dir += '/'
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # If it already exists, create a warning
    else:
        print(f"Warning: {save_dir} already exists. Files may be overwritten.")
        
    # Write the README.txt file containing the parameters used to generate the dataset
    with open(save_dir + 'README.txt', 'w') as f:
        f.write("Parameters used to generate the dataset:\n")
        for name, value in zip(param_names, param_values):
            f.write(f"{name}: {value}\n")
    
    # Create the dataset from the parameter combinations
    if n_cores == 1:
        for combo in tqdm(param_combinations):
            num_monomers, mean_bond_length, std_bond_length, num_templates, weight_dist, noise_std, num_observations, num_dataset, _ = combo
            
            dataset_list = []
            for idx_dataset in range(num_dataset):
                # Generate template chains 
                template_chain_list = [generate_gaussian_chain(num_monomers, mean_bond_length, std_bond_length) for i in range(num_templates)]
                
                # Generate observations based on weight distribution, num_observations, and noise std
                # Calculate the concentration parameter alpha based on weight_dist
                # When weight_dist is 0, alpha should be very small (resulting in more uneven numbers)
                # When weight_dist is 1, alpha should be large (resulting in more even numbers)
                alpha = np.maximum(np.ones(num_templates) * weight_dist * 100, np.ones(num_templates)) 
                num_observation_list = np.int32(np.random.dirichlet(alpha) * num_observations) + 1
                observation_list = [generate_observations(c, n, noise_std) for c, n in zip(template_chain_list, num_observation_list)]
                
                # Concatenate observation_list into a single array
                observation_list = np.concatenate([*observation_list])
                labels_true = np.concatenate([[i for _ in range(j)] for i, j in enumerate(num_observation_list)])
                
                dataset_list.append({
                        'template_chain_list': template_chain_list,
                        'observation_list': observation_list,
                        'labels': labels_true
                    })
                
            # Create a pickle file to save the dataset
            pickle_file = save_dir + f"dataset_{num_monomers}_{mean_bond_length}_{std_bond_length}_{num_templates}_{weight_dist}_{noise_std}_{num_observations}.pkl"
            tmp_file = pickle_file + '.tmp'
            
            with open(tmp_file, 'wb') as f:
                pickle.dump(dataset_list, f)
            os.rename(tmp_file, pickle_file)
    else:
        # Use parallel processing to generate the dataset
       
        # Prepare the arguments for the function
        args = [(combo, save_dir) for combo in param_combinations]
        
        # Using Pool with imap_unordered for progress tracking
        # Change n_cores to 5 so it does not OOM
        with Pool(5) as p:
            list(tqdm(p.imap_unordered(generate_dataset_parallel, args), total=len(args)))
            
    # Create a csv file for mapping parameter combos to pickle files
    with open(save_dir + 'parameter_mapping.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(param_names + ['pickle_file'])
        for combo in param_combinations:
            num_dataset = combo[7]
            for idx_dataset in range(num_dataset):
                pickle_file = save_dir + f"dataset_{combo[0]}_{combo[1]}_{combo[2]}_{combo[3]}_{combo[4]}_{combo[5]}_{combo[6]}_{combo[7]}.pkl"
                writer.writerow(list(combo) + [pickle_file])
    
    # Print a message to indicate that the dataset has been generated
    print(f"Dataset generated and saved at {save_dir}")
        

def load_dataset(pickle_file):
    """
    Load the dataset from the pickle file.

    Parameters
    ----------
    pickle_file : str
        The path to the pickle file containing the dataset.

    Returns
    -------
    tuple
        A tuple of the observation list, the true labels, and a dictionary of parameters.

    Raises
    ------
    FileNotFoundError
        If the pickle file does not exist.

    Notes
    -----
    This function loads the dataset from a pickle file. The pickle file should contain a tuple
    with the observation list and the true labels. It also extracts parameters from the file name
    and puts them in a dictionary.

    Example
    -------
    >>> pickle_file = 'data.pickle'
    >>> observation_list, labels_true, param_dict = load_dataset(pickle_file)
    """
    with open(pickle_file, 'rb') as f:
        dataset_list = pickle.load(f)

        
    # Extract parameters from file name
    # Remove .pkl file suffix
    pickle_file = pickle_file.replace('.pkl', '')
    params = os.path.basename(pickle_file).split('_')[1:]
    params = [int(p) if p.isdigit() else float(p) for p in params]

    # Put the parameters in a dictionary
    param_dict = {
        'num_monomers': params[0],
        'mean_bond_length': params[1],
        'std_bond_length': params[2],
        'num_templates': params[3],
        'weights_dist': params[4],
        'noise_std': params[5],
        'num_observations': params[6]
    }
        
    return dataset_list, param_dict


def logprior(dmap_flat, num_probes):
    """
    """
    return np.sum(np.array(_logprior(dmap_flat, num_probes)))


def _logprior(dmap_flat, num_probes):
    """
    """
    # Get 2D map back to simplify the expression 
    dmap = np.reshape(dmap_flat, [num_probes, num_probes])
    
    # Calculate square of all bond lengths
    R_sq = np.diag(dmap, 1) ** 2
    
    # Calculate the total number of Kuhn segments 
    # equals to the contour length divided by the Kuhn length
    contour_length = np.sum(np.diag(dmap, 1))
    
    # If contour length is zero or negative, or distance map contains negative values
    # return very low probability: the lowest number numpy can handle
    if contour_length <= 0 or np.any(dmap <= -1):
        # This will return the smallest negative number, to reflect the lowest probability
        return np.iinfo(np.int32).min, np.iinfo(np.int32).min
        
    N_kuhn = num_probes - 1 # number of Kuhn segments
    b = contour_length / num_probes  # Kuhn length
    
    # Calculate the probability
    scaling_factor = 1.5 * N_kuhn * np.log(3/(2*np.pi*b**2))
    gaussian_term = -3*np.sum(R_sq)/(2*b**2)
    
    return scaling_factor, gaussian_term 

## TODO: change the name measurement_error to pixel variance

# def loglikelihood(dmap_flat, ref_dmap_flat, measurement_error, num_probes):
#     """
#     """
#     return np.sum(np.array(_loglikelihood(dmap_flat, ref_dmap_flat, measurement_error, num_probes)))


# def _loglikelihood(dmap_flat, ref_dmap_flat, measurement_error, num_probes):
#     """ 
#     """
#     # if the reference distance map is not physical ie contains negative values
#     # return very low probability: the lowest number numpy can handle
#     if np.any(ref_dmap_flat <= -1):
#         return np.iinfo(np.int32).min, np.iinfo(np.int32).min
        
    
#     # Calculate the difference between distance map and reference 
#     # distance map
#     subtraction_map_sq = np.square(dmap_flat - ref_dmap_flat).reshape(num_probes, num_probes)

#     # Only consider the upper triangular part of the distance map
#     triu_indices = np.triu_indices(num_probes, k=1)
#     measurement_error = 2*measurement_error[triu_indices]  # both triangles 
#     subtraction_map_sq = 2*subtraction_map_sq[triu_indices]  # both triangles
    
#     # Calculate the normalization factor
#     normalization_factor = -np.sum(np.log(np.sqrt(2*np.pi*measurement_error**2)))
    
#     # Calculate the gaussian term 
#     gaussian_term = -np.sum(subtraction_map_sq/(2*np.square(measurement_error)))
    
#     return normalization_factor, gaussian_term


def loglikelihood(observed_dmap_flat, microstates_dmap_flat, measurement_error, num_probes):
    """ 
    """
    num_microstates = microstates_dmap_flat.shape[0]
    num_observations = observed_dmap_flat.shape[0]
    
    # Append a new axis for broadcasting
    observed_dmap_flat = observed_dmap_flat[np.newaxis, :, :]
    microstates_dmap_flat = microstates_dmap_flat[:, np.newaxis, :]
    measurement_error = measurement_error[:, np.newaxis, :, :]
        
    
    # Calculate the difference between distance map and reference 
    # distance map
    subtraction_map_sq = np.square(observed_dmap_flat - microstates_dmap_flat).reshape(num_microstates, num_observations, 
                                                                      num_probes, num_probes)

    # Only consider the upper triangular part of the distance map
    triu_indices = np.triu_indices(num_probes, k=1)
    measurement_error = 2*measurement_error[:, :, triu_indices[0], triu_indices[1]]  # both triangles 
    subtraction_map_sq = 2*subtraction_map_sq[:, :, triu_indices[0], triu_indices[1]]  # both triangles
    
    # Calculate the normalization factor
    normalization_factor = -np.sum(np.log(np.sqrt(2*np.pi*measurement_error**2)), axis=-1)
    
    # Calculate the gaussian term 
    gaussian_term = -np.sum(subtraction_map_sq/(2*np.square(measurement_error)), axis=-1)
    
    # if the reference distance map is not physical ie contains negative values
    # return very low probability: the lowest number numpy can handle
    if np.any(microstates_dmap_flat <= -1):
        unphysical_microstates_indices = np.any(microstates_dmap_flat < 0, axis=-1)
        normalization_factor[unphysical_microstates_indices] = np.iinfo(np.int32).min
        gaussian_term[unphysical_microstates_indices] = np.iinfo(np.int32).min
    
    # Change the dimension so it is compatible with the downstream analysis
    return np.transpose(normalization_factor + gaussian_term)


def interpolate_polymers(polys):
    """
    Interpolates missing values (NaNs) in the polymer data across multiple probes, coordinates, and cells.
    This function also removes cells with missing values in all probes.

    Parameters:
    polys (np.ndarray): A 3D numpy array with shape (num_probes, num_coords, num_cells) 
                        representing the polymer data. The array contains NaNs where data is missing.

    Returns:
    np.ndarray: A new 3D numpy array with the same shape as the input, but with missing values 
                interpolated along the probe dimension for each coordinate and cell.
    """
    # Extract the dimensions of the input array
    num_probes, num_coords, num_cells = polys.shape
    
    # Initialize an array of the same shape to hold the interpolated values
    new_polys = np.zeros((num_probes, num_coords, num_cells))
    
    # Iterate over each cell
    for c in range(num_cells):
        # Extract the data for the current cell
        curr_cells = polys[:, :, c]
        
        # Skip cells with all missing values
        if np.all(np.isnan(curr_cells)):
            continue # This leaves a matrix of zeros in the output array
        
        # Iterate over each coordinate
        for x in range(num_coords):
            # Extract the data for the current coordinate across all probes
            curr_coords = curr_cells[:, x]
            
            # Identify the indices of missing (NaN) values
            missing_indices = np.isnan(curr_coords)
            
            # Identify the indices of valid (non-NaN) values
            valid_indices = ~missing_indices
            
            # Interpolate missing values based on the valid values
            interp_coords = np.interp(np.flatnonzero(missing_indices), 
                                      np.flatnonzero(valid_indices), 
                                      curr_coords[valid_indices])
            
            # Assign the interpolated values to the corresponding positions in the output array
            new_polys[missing_indices, x, c] = interp_coords
            
            # Copy the valid (non-NaN) values to the output array
            new_polys[valid_indices, x, c] = curr_coords[valid_indices]
    
    # Return the array with interpolated values
    return new_polys


def calculate_distance_map(polys):
    """
    Compute the pairwise Euclidean distance maps for each cell in a 3D input array.

    The function takes an input array `polys` representing coordinate data for different probes 
    across multiple cells, computes the pairwise Euclidean distance between probes within each 
    cell, and returns a 3D array where each slice corresponds to the distance map for one cell.

    Parameters:
    -----------
    polys : numpy.ndarray
        A 3D array of shape (num_probes, num_coords, num_cells) where:
        - `num_probes` is the number of probes.
        - `num_coords` is the dimensionality of the coordinates (e.g., 2 for 2D, 3 for 3D).
        - `num_cells` is the number of cells.

    Returns:
    --------
    new_maps : numpy.ndarray
        A 3D array of shape (num_cells, num_probes, num_probes), where each slice (along the first axis)
        is a matrix containing the pairwise Euclidean distances between probes for a given cell.
        Cells with all missing values (NaNs) will result in zero matrices.

    Notes:
    ------
    - If all coordinate data for a cell are missing (NaN values), the corresponding distance map 
      will be left as a zero matrix in the output.
    - The function uses the `pdist` function from `scipy.spatial.distance` to compute pairwise distances.
    """
    # Extract the dimensions of the input array
    num_probes, num_coords, num_cells = polys.shape
    
    # Initialize an array of the same shape to hold the interpolated values
    new_maps = np.zeros((num_cells, num_probes, num_probes))
    
    # Iterate over each cell
    for c in range(num_cells):
        # Extract the data for the current cell
        curr_cells = polys[:, :, c]
        
        # Skip cells with all missing values
        if np.all(np.isnan(curr_cells)):
            continue  # This leaves a matrix of zeros in the output array
        
        # Calculate the pairwise Euclidean distance between each pair of probes
        dmap = squareform(pdist(curr_cells))
        
        # Assign the distance map to the corresponding position in the output array
        new_maps[c, :, :] = dmap
    
    # Return the array with interpolated values
    return new_maps


def calculate_conformational_variance(dmap_list, dmap_ref):
    """
    Calculate the conformational variation of a set of distance maps relative to a reference map.

    Parameters:
    dmap_list (list): A list of 2D numpy arrays representing the distance maps.
    dmap_ref (np.ndarray): A 2D numpy array representing the reference distance map.
    num_probes (int): The number of probes in the distance maps.

    Returns:
    np.ndarray: A 2D numpy array containing the variance of the squared Euclidean distances 
               between each distance map and the reference map.
    """
    # Convert dmap_list to a NumPy array
    dmap_list = np.array(dmap_list)
    
    # Calculate the squared Euclidean distance between each distance map and the reference map
    diff_list = np.sqrt((dmap_list - dmap_ref) ** 2) 
    
    # Calculate the variance along the number of observation/cell dimension
    var = np.var(diff_list, axis=0)
    
    return var


def calculate_distance_map(polys):
    # Extract the dimensions of the input array
    num_probes, num_coords, num_cells = polys.shape
    
    # Initialize an array of the same shape to hold the interpolated values
    new_maps = np.zeros((num_cells, num_probes, num_probes))
    
    # Iterate over each cell
    for c in range(num_cells):
        # Extract the data for the current cell
        curr_cells = polys[:, :, c]
        
        # Skip cells with all missing values
        if np.all(np.isnan(curr_cells)):
            continue  # This leaves a matrix of zeros in the output array
        
        # Calculate the pairwise Euclidean distance between each pair of probes
        dmap = squareform(pdist(curr_cells))
        
        # Assign the distance map to the corresponding position in the output array
        new_maps[c, :, :] = dmap
    
    # Return the array with interpolated values
    return new_maps


def calculate_conformational_variance_new(dmap_list, microstates_dmap):
    # This is incorrect because it finds mean across all samples 
    """
    Calculate the conformational variation of a set of distance maps relative to a reference map.

    Parameters:
    dmap_list (list): A list of 2D numpy arrays representing the distance maps.
    dmap_ref (np.ndarray): A 2D numpy array representing the reference distance map.
    num_probes (int): The number of probes in the distance maps.

    Returns:
    np.ndarray: A 2D numpy array containing the variance of the squared Euclidean distances 
               between each distance map and the reference map.
    """
    # Convert dmap_list to a NumPy array
    dmap_list = np.array(dmap_list)
    
    num_microstates = microstates_dmap.shape[0]
    num_probes = np.round(microstates_dmap.shape[1] ** 0.5).astype(int)
    
    dmap_list = dmap_list[:, np.newaxis, :]
    microstates_dmap = microstates_dmap[np.newaxis, :, :]
    
    # Calculate the squared Euclidean distance between each distance map and the reference map
    diff_list = np.sqrt((dmap_list - microstates_dmap) ** 2)
    
    # Calculate the variance along the number of observation/cell dimension
    var = np.var(diff_list, axis=0)
    
    return np.reshape(var, (num_microstates, num_probes, num_probes))


def load_weights(directory, num_metastructures):
    log_weights = []
    lp = []
    files = sorted(os.listdir(directory))[-4:]
    print(files)
    
    log_weights_d = []
    for file in files:
        log_weights_chain = []
        lp_chain = []
        with open('%s/%s'%(directory, file), newline='') as csvfile:
            reader = csv.DictReader(filter(lambda row: row[0]!='#', csvfile), )
            for row in reader:
                log_weights_row = [float(row["log_weights.%d"%i]) for i in range(1,num_metastructures+1)]
                lp_chain.append(float(row["lp__"]))
                log_weights_chain.append(log_weights_row)
        log_weights = np.array(log_weights_chain)
        lp_chain = np.array(lp_chain)
        log_weights_d.append(log_weights)
        lp.append(lp_chain)
    log_weights_d = np.array(log_weights_d)
    return log_weights_d 






