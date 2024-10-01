import pickle 
import sys

# add path to my polychrom installation 
# sys.path.append(r"/mnt/ceph/users/tudomlumleart/00_VirtualEnvironments/teeu/lib/python3.10/site-packages")
# sys.path.append(r"/mnt/home/tudomlumleart/.local/lib/python3.10/site-packages/")
# sys.path.append(r"/mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jupyter-gpu/lib/python3.10/site-packages")
import os 
import h5py
import ast 
import math
import copy
# import openmm
import jax 
import jaxopt
import scipy
import mpltern
import shutil
import multiprocessing
import umap
import time 
import seaborn as sns
import pandas as pd 
import extrusion1Dv2 as ex1D
import jax.numpy as jnp
import jax.scipy as jscipy
import jax.random as random
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm.auto import tqdm
from scipy.optimize import minimize 
from scipy import spatial 
# from LEBondUpdater import bondUpdater
from jax import grad, jit, vmap
from jax import random
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import pdist, squareform
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_simplex, projection_non_negative

# add path to my polychrom installation 
# sys.path.append(r"/mnt/home/tudomlumleart/.local/lib/python3.10/site-packages/")

# import polychrom
# from polychrom.starting_conformations import grow_cubic
# from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file
# from polychrom.simulation import Simulation
# from polychrom import polymerutils
# from polychrom import forces
# from polychrom import forcekits

jax.config.update("jax_enable_x64", True)

def generate_gaussian_chain(num_monomers: int, 
                            mean_bond_length: float, 
                            std_bond_length: float):
    """Generate a Gaussian chain polymer 
    
    Parameters
    ----------
    num_monomers
    mean_bond_length
    std_bond_length
    
    Return
    ------
    np.array 
    
    Notes
    -----
    """ 
    # Generate steps: each step is a 3D vector 
    steps = np.random.normal(mean_bond_length, std_bond_length, size=(num_monomers, 3))

    # Compute positions by cumulative sum of steps
    positions = np.cumsum(steps, axis=0)
    
    return positions


def visualize_polymer(polymer_chain, save_path=''):
    """Plot a polymer chain in 3D space
    
    Parameters
    ----------
    polymer_chain
    
    Return
    ------
    
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

    
def compare_distance_maps(chain1, chain2, type1='polymer', type2='polymer'):
    """ Plot distance maps of chain1 and chain2 side-by-side
    
    Parameters
    ----------
    chain1
    chain2 
    type1
    type2

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
    plt.show()
    
    
def generate_flatten_distance_map(chain):
    """ 
    """
    return jnp.reshape(squareform(pdist(chain)), -1)


# Plot the comparison between template chain and noisy chain
def compare_polymer_chains(chain1, chain2): 
    """ Visualize chain1 and chain2 side-by-side
    
    Parameters
    ----------
    chain1
    chain2 

    """
    # Initialize new figure
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    axs = [ax1, ax2]
    
    # Put polymer chains in the same data structure 
    chains = [chain1, chain2]
    
    # Plot polymer chain
    for poly_num, ax in enumerate(axs):
        polymer_chain = chains[poly_num]
        
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
        
        
        # Plot using scatter
        ax.plot(x_fine, y_fine, z_fine, 'gray', label='Interpolated Path')
        for i in range(num_monomers):
            ax.scatter(x[i], y[i], z[i], color=monomer_colors[i], s=50, alpha=0.75) 

        # Labeling the axes (optional but recommended for clarity)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    plt.show()
    
    
def generate_observations(polymer_chain, num_observations, gaussian_noise_std):
    """ Given a template polymer chain, generate num_observations polymer chains by adding 
    some gaussian noise to the polymer chain
    
    Parameters
    ----------
    polymer_chain
    num_observations
    gaussian_noise_std
    
    Return
    ------
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


@jax.jit
def likelihood(dmap_flat, ref_dmap_flat, measurement_error, num_probes):
    """ 
    """
    return jnp.prod(jnp.array(likelihood_(dmap_flat, ref_dmap_flat, measurement_error, num_probes)))


@jax.jit
def likelihood(dmap_flat, ref_dmap_flat, measurement_error, num_probes):
    """ 
    """
    return jnp.prod(jnp.array(likelihood_(dmap_flat, ref_dmap_flat, measurement_error, num_probes)))


@jax.jit
def loglikelihood(dmap_flat, ref_dmap_flat, measurement_error, num_probes):
    """
    """
    return jnp.sum(jnp.array(loglikelihood_(dmap_flat, ref_dmap_flat, measurement_error, num_probes)))


@jax.jit
def loglikelihood_(dmap_flat, ref_dmap_flat, measurement_error, num_probes):
    """ 
    """
    # Calculate the difference between distance map and reference 
    # distance map
    subtraction_map_sq = jnp.square(dmap_flat - ref_dmap_flat)
    sum_subtraction_map_sq = jnp.sum(subtraction_map_sq)
    
    # Calculate the normalization factor
    normalization_factor = -jnp.square(num_probes) * jnp.log(jnp.sqrt(2*np.pi*jnp.square(measurement_error)))
    
    # Calculate the gaussian term 
    gaussian_term = -jnp.sum(sum_subtraction_map_sq)/(2*jnp.square(measurement_error))
    
    # print('Scaling factor = {}'.format(normalization_factor))
    # print('Gaussian term = {}'.format(gaussian_term))
    
    return normalization_factor, gaussian_term

@partial(jax.jit, static_argnums=1)
def prior(dmap_flat, num_probes):
    """
    """
    return jnp.prod(jnp.array(prior_(dmap_flat, num_probes)))

@partial(jax.jit, static_argnums=1)
def prior_(dmap_flat, num_probes):
    """
    """
    # Get 2D map back to simplify the expression 
    dmap = jnp.reshape(dmap_flat, [num_probes, num_probes])
    
    # Calculate the squared end-to-end distance 
    R_sq = dmap[0][-1] ** 2
    
    # Calculate the average bond length
    b = jnp.mean(jnp.diag(dmap, 1))
    
    N = num_probes
    
    # Calculate the probability
    scaling_factor = (3/(2*np.pi*N*b**2)) ** 1.5
    gaussian_term = jnp.exp(-3*R_sq/(2*N*b**2))
    
    # print('Scaling factor = {}'.format(scaling_factor))
    # print('Gaussian term = {}'.format(gaussian_term))
    
    return scaling_factor, gaussian_term 

def logprior(dmap_flat, num_probes):
    """
    """
    return jnp.sum(jnp.array(logprior_(dmap_flat, num_probes)))

@partial(jax.jit, static_argnums=(1, ))
def logprior_(dmap_flat, num_probes):
    """
    """
    # Get 2D map back to simplify the expression 
    dmap = jnp.reshape(dmap_flat, [num_probes, num_probes])
    
    # Calculate the squared end-to-end distance 
    R_sq = dmap[0][-1] ** 2
    
    # Calculate the average bond length
    b = jnp.mean(jnp.diag(dmap, 1))
    
    N = num_probes
    
    # Calculate the probability
    scaling_factor = 1.5 * jnp.log(3/(2*np.pi*N*b**2))
    gaussian_term = -3*R_sq/(2*N*b**2)
    
    # print('Scaling factor = {}'.format(scaling_factor))
    # print('Gaussian term = {}'.format(gaussian_term))
    
    return scaling_factor, gaussian_term 

def visualize_dmap(dmap, vmax=None, save_path=''):
    """Plot a distance map 
    
    Parameters
    ----------
    
    """
    # Create a new matplotlib figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
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
    
    
def generate_posterior_parallelize(templates, observations, template_weights, weight_renormalization=1000):
    """
    """
    templates_flatten = jnp.array([generate_flatten_distance_map(t) for t in templates])
    observations_flatten = jnp.array([generate_flatten_distance_map(o) for o in observations])
    template_weights = jnp.array(template_weights)
    
    weight_prior = 1/len(template_weights) 
    
    # Generate grid index combination
    template_info_indices = jnp.arange(len(templates_flatten))
    observation_info_indices = jnp.arange(len(observations_flatten))
    t_ind, o_ind = jnp.meshgrid(template_info_indices, observation_info_indices)
    
    total_posterior = 0
    
    t_ind = t_ind.flatten()
    o_ind = o_ind.flatten()
    
    jax.debug.print("Weights at current iteration: {y}", y=template_weights)
    def calculate_rhs(t_ind, o_ind):
        val = 0 
        o = observations_flatten[o_ind]
        t = templates_flatten[t_ind]
        alpha = template_weights[t_ind]
        
        val += loglikelihood(o, t, measurement_error, num_probes)

        val += logprior(t, num_probes)

        # This is the correct one 
        # But the scaling between alpha and weight priors and logliokelihood are so different 
        # val += jnp.log(alpha + 1e-32) * weight_renormalization 
        val += jnp.log(jnp.abs(alpha) + 1e-32) * weight_renormalization  # use jnp.abs to make sure that each alpha does not go to 0
        val += jnp.log(weight_prior) * weight_renormalization
           
        return val 
    
    def calculate_posterior(i):
        return jscipy.special.logsumexp(jnp.where(o_ind == i, curr_obs_list, -jnp.inf))
    
    curr_obs_list = jnp.array(jax.vmap(calculate_rhs)(t_ind, o_ind))
    total_posterior = jnp.sum(jax.vmap(calculate_posterior)(jnp.arange(len(observations))))

    return total_posterior


def weight_neg_objective_parallelize(template_weights):
    """
    """
    templates = template_chain_list
    observations = observation_list
    return -generate_posterior_parallelize(templates, observations, template_weights)
    