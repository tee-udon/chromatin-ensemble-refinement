import jax 
import jaxopt
import scipy
import mpltern
import jax.numpy as jnp
import jax.scipy as jscipy
import jax.random as random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from scipy.optimize import minimize 
from jax import grad, jit, vmap
from jax import random
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import pdist, squareform

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
    
    # Determine the scale of colorbars
    # Both colorbars show data in the same range 
    cm_min = 0
    cm_max = np.max([np.max(distance_map1), np.max(distance_map2)])
    
    # Initialize new figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot distance maps 
    heatmap1 = ax1.imshow(distance_map1, cmap='hot', aspect='auto', vmin=cm_min, vmax=cm_max)
    ax1.set_title('Chain 1')
    cb1 = fig.colorbar(heatmap1, ax=ax1)
    cb1.set_label('Euclidean distance [a.u.]')
    
    heatmap2 = ax2.imshow(distance_map2, cmap='hot', aspect='auto', vmin=cm_min, vmax=cm_max)
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


@jit
def likelihood(dmap_flat, ref_dmap_flat, measurement_error, num_probes):
    """ 
    """
    return jnp.prod(jnp.array(likelihood_(dmap_flat, ref_dmap_flat, measurement_error, num_probes)))


@jit
def likelihood(dmap_flat, ref_dmap_flat, measurement_error, num_probes):
    """ 
    """
    return jnp.prod(jnp.array(likelihood_(dmap_flat, ref_dmap_flat, measurement_error, num_probes)))


@jit
def loglikelihood(dmap_flat, ref_dmap_flat, measurement_error, num_probes):
    """
    """
    return jnp.sum(jnp.array(loglikelihood_(dmap_flat, ref_dmap_flat, measurement_error, num_probes)))


@jit
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


def prior(dmap_flat, num_probes):
    """
    """
    return jnp.prod(jnp.array(prior_(dmap_flat, num_probes)))


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

