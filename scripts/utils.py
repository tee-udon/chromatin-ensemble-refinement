import sys
sys.path.append(r"/mnt/home/tudomlumleart/.local/lib/python3.10/site-packages/")
sys.path.append(r"/mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jupyter-gpu/lib/python3.10/site-packages")
sys.path.append(r"/mnt/ceph/users/tudomlumleart/00_VirtualEnvironments/teeu/lib/python3.10/site-packages")

import os 
import h5py
import ast 
import math
import copy
import csv 
# import jax 
# import jaxopt
import scipy
import mpltern
import shutil
import multiprocessing
import torch
import umap
import time 
import sklearn
import numpy 
import pickle
import itertools
import seaborn as sns
import pandas as pd 
# import jax.numpy as jnp
# import jax.scipy as jscipy
# import jax.random as random
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from numpy.typing import ArrayLike
from typing import Literal
from sklearn.mixture import BayesianGaussianMixture
from scipy.optimize import linear_sum_assignment
from functools import partial
from tqdm.auto import tqdm
from scipy.optimize import minimize 
from scipy import spatial 
# from jax import grad, jit, vmap
# from jax import random
from sklearn.metrics import adjusted_mutual_info_score
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import pdist, cdist, squareform
# from jaxopt import ProjectedGradient
# from jaxopt.projection import projection_simplex, projection_non_negative


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
        
    
# Test load dataset and extract parameters 
# Train on bgmm with specific hyperparameters
# Evaluate performances using score and loss functions defined above 

# First define the function to assign clusters to the predicted weights
# based on the minimum distance between true_means and pred_means
def assign_clusters(mean_true, mean_pred, index=False):
    """
    Assigns clusters to the true and predicted means based on the minimum distance between them.

    Parameters
    ----------
    mean_true : numpy.ndarray
        Array of true means.
    mean_pred : numpy.ndarray
        Array of predicted means.
    index : bool, optional
        Flag indicating whether to return the row and column indices of the sorted mean_true and mean_pred. Default is False.

    Returns
    -------
    tuple
        If index is False, returns a tuple of sorted mean_true and sorted mean_pred based on minimum distance.
        If index is True, returns a tuple of row and column indices of the sorted mean_true and mean_pred.
    """

    # Calculate the pairwise distance between the true and predicted means
    pred_error_mat = cdist(mean_pred, mean_true)

    # Assign the predicted weights to the true weights based on the minimum distance
    row_ind, col_ind = linear_sum_assignment(pred_error_mat)

    # if index flag is False, return the sorted mean_true and sorted mean_pred based on minimum distance
    if not index:
        return mean_true[col_ind], mean_pred[row_ind]
    # if index flag is True, return the row and column indices of the sorted mean_true and mean_pred
    else:
        return col_ind, row_ind

def MSE_means(mean_true_assigned, mean_pred_assigned):
    """
    Calculate the mean squared error between the true and predicted means.
    
    These true and predicted means have been assigned cluster based on the pairwise distance between them.

    Parameters
    ----------
    mean_true_assigned : array-like
        The true means assigned.
    mean_pred_assigned : array-like
        The predicted means assigned.

    Returns
    -------
    mse : float
        The mean squared error between the true and predicted means.
        
    See Also
    --------
    assign_clusters : Assigns clusters to the true and predicted means based on the minimum distance between them.

    """

    # Ensure that the mean_true and mean_pred are numpy arrays
    mean_true = np.array(mean_true_assigned)
    mean_pred = np.array(mean_pred_assigned)
    
    # Calculate the mean squared error between the true and predicted means
    mse = np.mean((mean_true_assigned - mean_pred_assigned)**2, axis=(1, 0))
    
    return mse

# KL divergence for weights 
def KL_div_weights(weights_true_assigned, weights_pred_assigned):
    """
    Calculates the Kullback-Leibler (KL) divergence between the true and predicted weights.

    Parameters
    ----------
    weights_true : ArrayLike of float
        The true weights of the components. These true weights have already been assigned groups.
    weights_pred : ArrayLike of float
        The predicted weights of the components. These predicted weights have already been assigned groups.
        
    Returns
    -------
    kl_div (float): The KL divergence between the true and predicted weights.
    """
    
    # Ensure that the weights_true and weights_pred are numpy arrays
    weights_true = np.array(weights_true_assigned)
    weights_pred = np.array(weights_pred_assigned)
    
    # Calculate the KL divergence between the true and predicted weights
    kl_div = np.sum(weights_true_assigned * np.log(weights_true_assigned / weights_pred_assigned))
    
    return kl_div

# Frobenius norm for covariance matrices
def MSD_covariances(covariances_true_assigned, covariances_pred_assigned):
    """
    Calculate the Frobenius norm of the difference between the true and predicted covariances.
    
    These true and predicted covariances have been assigned cluster based on the pairwise distance between them.

    Parameters
    ----------
    covariances_true_assigned : array-like
        The flattened true covariances assigned.
    covariances_pred_assigned : array-like
        The flattened predicted covariances assigned.

    Returns
    -------
    mse : float
        The mean squared error between the true and predicted covariances.
        
        
    See Also
    --------
    assign_clusters : Assigns clusters to the true and predicted covariances based on the minimum distance between them.

    """
    # Ensure that the covariances_true and covariances_pred are numpy arrays
    covariances_true_assigned = np.array(covariances_true_assigned)
    covariances_pred_assigned = np.array(covariances_pred_assigned)
    
    # Calculate the mean squared error between the true and predicted covariances
    mse = np.mean((covariances_true_assigned - covariances_pred_assigned)**2, axis=(1, 0))
    
    return mse

def corr_covariances(covariances_true_assigned, covariances_pred_assigned):
    """
    Calculate the correlation between the true and predicted covariances.

    These true and predicted covariances have been assigned cluster based on the pairwise distance between them.

    Parameters
    ----------
    covariances_true_assigned : array-like
        The flattened true covariances assigned.
    covariances_pred_assigned : array-like
        The flattened predicted covariances assigned.

    Returns
    -------
    corr : float
        The correlation between the true and predicted covariances.
        
        
    See Also
    --------
    assign_clusters : Assigns clusters to the true and predicted covariances based on the minimum distance between them.

    """
    # Ensure that the covariances_true and covariances_pred are numpy arrays
    covariances_true_assigned = np.array(covariances_true_assigned)
    covariances_pred_assigned = np.array(covariances_pred_assigned)
    
    # Calculate the correlation between the true and predicted covariances
    corr = np.corrcoef(covariances_true_assigned, covariances_pred_assigned)[0, 1]
    
    return corr

def corr_weights(weights_true_assigned, weights_pred_assigned):
    """
    Calculate the correlation between the true and predicted weights.
    
    These true and predicted weights have been assigned cluster based on the pairwise distance between them.

    Parameters
    ----------
    weights_true_assigned : array-like
        The true weights assigned.
    weights_pred_assigned : array-like
        The predicted weights assigned.

    Returns
    -------
    corr : float
        The correlation between the true and predicted weights.
        
        
    See Also
    --------
    assign_clusters : Assigns clusters to the true and predicted weights based on the minimum distance between them.

    """
    # Ensure that the weights_true and weights_pred are numpy arrays
    weights_true_assigned = np.array(weights_true_assigned)
    weights_pred_assigned = np.array(weights_pred_assigned)
    
    # Calculate the correlation between the true and predicted weights
    corr = np.corrcoef(weights_true_assigned, weights_pred_assigned)[0, 1]
    
    return corr