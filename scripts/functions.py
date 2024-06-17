import pickle 
import sys

# add path to my polychrom installation 
sys.path.append(r"/mnt/home/tudomlumleart/.local/lib/python3.10/site-packages/")
sys.path.append(r"/mnt/home/tudomlumleart/00_VirtualEnvironments/jupyter-gpu/lib/python3.10/site-packages")

import os 
import h5py
import ast 
import math
import copy
import openmm
import jax 
import jaxopt
import scipy
import mpltern
import shutil
import multiprocessing
import torch
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
from LEBondUpdater import bondUpdater
from jax import grad, jit, vmap
from jax import random
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import pdist, squareform
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_simplex, projection_non_negative

# add path to my polychrom installation 
sys.path.append(r"/mnt/home/tudomlumleart/.local/lib/python3.10/site-packages/")

import polychrom
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file
from polychrom.simulation import Simulation
from polychrom import polymerutils
from polychrom import forces
from polychrom import forcekits

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

@partial(jax.jit, static_argnums=1)
def logprior(dmap_flat, num_probes):
    """
    """
    return jnp.sum(jnp.array(logprior_(dmap_flat, num_probes)))

@partial(jax.jit, static_argnums=1)
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

def generate_polymer_chain(
    num_monomers: int, 
    num_polymers: int,
    num_observations: int, 
    save_folder: str, 
    # optional arguments 
    monomer_types: np.ndarray = None, # make sure that len(monomer_type_list) == num_monomers 
    interaction_matrix: np.ndarray = None, # make sure dim = monomer_types x monomer_types 
    ctcf_sites: list = [],
    ctcf_directions: list = None, # make sure len(dir) == len(sites)
    ctcf_stall_probs: list = None, 
    num_lef: int = None, 
    lef_load_prob: float = None , 
    extra_bond_pairs: list = [], # make sure it is nested 
    num_templates: int = 100
    ): 
    """ 
    Add documentation here
    """
    if save_folder[-1] != '/':
        save_folder += '/'
    
    # Initialize optional arguments 
    if monomer_types is None:
        monomer_types = np.zeros(num_monomers).astype(int)
    assert len(monomer_types) == num_monomers, "The length of monomer_types types should equal to num_monomers!"
    
    num_unique_monomer_type = len(np.unique(monomer_types))
    if interaction_matrix is None:
        interaction_matrix = np.zeros([num_unique_monomer_type, num_unique_monomer_type]).astype(int)
    assert interaction_matrix.shape == (num_unique_monomer_type, num_unique_monomer_type), \
            "The dimension of interaction matrix should equal to the number of unique monomer types x the number of unique monomer types!"
    
    if ctcf_directions is None:
        if len(ctcf_sites) == 0:
            ctcf_directions = []
        else:
            ctcf_directions = np.zeros(len(ctcf_sites)).astype(int)
    assert len(ctcf_directions) == len(ctcf_sites), "The number of CTCF directions should equal to the number of CTCF sites!"
    
    
    if lef_load_prob is None:
        lef_load_prob = np.tile(np.ones([1, num_monomers]), [1, num_polymers])
        lef_load_prob = lef_load_prob / np.sum(lef_load_prob)
    
    # Simulation parameters 
    density = 0.002  # density of the PBC box 
    N1 = num_monomers  # Number of monomers in the polymer
    M = num_polymers  # Number of separate chains in the same volume 
    N = N1 * M # Number of monomers in the full simulation 
    LIFETIME = 200  # [Imakaev/Mirny use 200 as demo] extruder lifetime
    SEPARATION = 10  # Average separation between extruders in monomer units
    ctcfSites = ctcf_sites
    nCTCF = np.shape(ctcfSites)[0]
    ctcfDir = ctcf_directions  # 0 is bidirectional, 1 is right 2 is left
    if ctcf_stall_probs is None:
        ctcfCapture = 0.99 * np.ones(nCTCF)  # capture probability per block if capture < than this, capture
        ctcfRelease = 0.01 * np.ones(nCTCF)  # release probability per block. if capture < than this, release
    else:
        ctcf_stall_probs = np.array(ctcf_stall_probs)
        ctcfCapture = ctcf_stall_probs
        ctcfRelease = 1 - ctcf_stall_probs
    assert len(ctcfCapture) == nCTCF, 'the length of ctcfCapture should equal to nCTCF!'
    assert len(ctcfRelease) == nCTCF, 'the length of ctcfRelease should equal to nCTCF!'
    
    oneChainMonomerTypes = monomer_types
    interactionMatrix = interaction_matrix
    loadProb = lef_load_prob
    print(loadProb)
    
    if num_lef is None:
        if len(ctcf_sites) == 0:
            num_lef = 0
        else:
            num_lef = num_monomers // SEPARATION
            
    LEFNum = num_lef
    monomers = N1
    
    # less common parameters
    attraction_radius = 1.5  # try making this larger; I might have to change repulsion radius too 
    num_chains = M  # simulation uses some equivalent chains  (5 in a real sim)
    MDstepsPerCohesinStep = 800
    smcBondWiggleDist = 0.2
    smcBondDist = 0.5
    angle_force = 1.5  # most sims ran with 1.5.  0 might have been better
    
    # save pars
    saveEveryBlocks = 100  # save every 10 blocks
    numObservations = num_observations
    restartSimulationEveryBlocks = numObservations * saveEveryBlocks # blocks per iteration
    trajectoryLength = 100  # Let the 1D simulation runs for 100 timesteps
    
    # check that these loaded alright
    print(f'LEF count: {LEFNum}')
    print('interaction matrix:')
    print(interactionMatrix)
    print('monomer types:')
    print(oneChainMonomerTypes)
    print(save_folder)

    # generate a new folder  
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    else:
        shutil.rmtree(save_folder)
        os.makedirs(save_folder)

    for i_template in tqdm(range(num_templates)):
        simulation_save_folder = save_folder + 'template_{}/'.format(i_template)
        
        if not os.path.isdir(simulation_save_folder):
            os.makedirs(simulation_save_folder)
        else:
            shutil.rmtree(simulation_save_folder)
            os.makedirs(simulation_save_folder)
        
        newFolder = simulation_save_folder
        lefPosFile = simulation_save_folder + "LEFPos.h5"
        print(lefPosFile)
        # remove previous LEFPos.h5 file
        if os.path.isfile(lefPosFile):
            os.remove(lefPosFile)
            
        # generate a new folder  
        if not os.path.isdir(newFolder):
            os.makedirs(newFolder)
        else:
            shutil.rmtree(newFolder)
            os.makedirs(newFolder)
            
        reporter = HDF5Reporter(folder=newFolder, max_data_length=100, check_exists=False)
        print('creating folder')
        
        # ==================================#
        # Run and load 1D simulation
        # =================================#
        
        ctcfLeftRelease = {}
        ctcfRightRelease = {}
        ctcfLeftCapture = {}
        ctcfRightCapture = {}
        
        # should modify this to allow directionality
        for i in range(M):  # loop over chains (this variable needs a better name Max)
            for t in range(len(ctcfSites)):
                print(ctcfSites)
                pos = i * N1 + ctcfSites[t]
                
                if ctcfDir[t] == 0:
                    ctcfLeftCapture[pos] = ctcfCapture[t]  # if random [0,1] is less than this, capture
                    ctcfLeftRelease[pos] = ctcfRelease[t]  # if random [0,1] is less than this, release
                    ctcfRightCapture[pos] = ctcfCapture[t]
                    ctcfRightRelease[pos] = ctcfRelease[t]
                elif ctcfDir[t] == 1:  # stop Cohesin moving toward the right
                    ctcfLeftCapture[pos] = 0
                    ctcfLeftRelease[pos] = 1
                    ctcfRightCapture[pos] = ctcfCapture[t]
                    ctcfRightRelease[pos] = ctcfRelease[t]
                elif ctcfDir[t] == 2:
                    ctcfLeftCapture[pos] = ctcfCapture[t]  # if random [0,1] is less than this, capture
                    ctcfLeftRelease[pos] = ctcfRelease[t]  # if random [0,1] is less than this, release
                    ctcfRightCapture[pos] = 0
                    ctcfRightRelease[pos] = 1
            
        args = {}
        args["ctcfRelease"] = {-1: ctcfLeftRelease, 1: ctcfRightRelease}
        args["ctcfCapture"] = {-1: ctcfLeftCapture, 1: ctcfRightCapture}
        args["N"] = N
        args["LIFETIME"] = LIFETIME
        args["LIFETIME_STALLED"] = LIFETIME  # no change in lifetime when stalled
        
        occupied = np.zeros(N)
        occupied[0] = 1  # (I think this is just prevent the cohesin loading at the end by making it already occupied)
        occupied[-1] = 1  # [-1] is "python" for end
        cohesins = []
        
        print('starting simulation with N LEFs=')
        print(LEFNum)
        for i in range(LEFNum):
            ex1D.loadOneFromDist(cohesins, occupied, args, loadProb)  # load the cohesins
        
        with h5py.File(lefPosFile, mode='a') as myfile:
            dset = myfile.create_dataset("positions",
                                            shape=(trajectoryLength, LEFNum, 2),
                                            dtype=np.int32,
                                            compression="gzip")
            steps = 100  # saving in 50 chunks because the whole trajectory may be large
            bins = np.linspace(0, trajectoryLength, steps, dtype=int)  # chunks boundaries
            for st, end in zip(bins[:-1], bins[1:]):
                cur = []
                for i in range(st, end):
                    ex1D.translocate(cohesins, occupied, args, loadProb)  # actual step of LEF dynamics
                    positions = [(cohesin.left.pos, cohesin.right.pos) for cohesin in cohesins]
                    cur.append(positions)  # appending current positions to an array
                cur = np.array(cur)  # when we finished a block of positions, save it to HDF5
                dset[st:end] = cur
            myfile.attrs["N"] = N
            myfile.attrs["LEFNum"] = LEFNum
        
        # =========== Load LEF simulation ===========#
        trajectory_file = h5py.File(lefPosFile, mode='r')
        LEFNum = trajectory_file.attrs["LEFNum"]  # number of LEFs
        LEFpositions = trajectory_file["positions"]  # array of LEF positions
        steps = MDstepsPerCohesinStep  # MD steps per step of cohesin  (set to ~800 in real sims)
        Nframes = LEFpositions.shape[0]  # length of the saved trajectory (>25000 in real sims)
        print(f'Length of the saved trajectory: {Nframes}')
        block = 0  # starting block
        
        # test some properties
        # assertions for easy managing code below
        # assert (Nframes % restartSimulationEveryBlocks) == 0
        assert (restartSimulationEveryBlocks % saveEveryBlocks) == 0
        
        savesPerSim = restartSimulationEveryBlocks // saveEveryBlocks
        simInitsTotal = (Nframes) // restartSimulationEveryBlocks
        # concatinate monomers if needed
        if len(oneChainMonomerTypes) != N:
            monomerTypes = np.tile(oneChainMonomerTypes, num_chains)
        else:
            monomerTypes = oneChainMonomerTypes
        
        N_chain = len(oneChainMonomerTypes)
        N = len(monomerTypes)
        print(f'N_chain: {N_chain}')  # ~8000 in a real sim
        print(f'N: {N}')  # ~40000 in a real sim
        N_traj = trajectory_file.attrs["N"]
        print(f'N_traj: {N_traj}')
        assert N == trajectory_file.attrs["N"]
        print(f'Nframes: {Nframes}')
        print(f'simInitsTotal: {simInitsTotal}')
        
        # ==============================================================#
        #                  RUN 3D simulation                              #
        # ==============================================================#
        # Initial simulation using fixed input states
        num_timepoint, num_lefs, _ = LEFpositions.shape
            
        # generate a new folder  
        if not os.path.isdir(simulation_save_folder):
            os.makedirs(simulation_save_folder)
        else:
            shutil.rmtree(simulation_save_folder)
            os.makedirs(simulation_save_folder)

        LEFsubset = LEFpositions[num_timepoint-1:num_timepoint, :, :]  # a subset of the total LEF simulation time
        milker = bondUpdater(LEFsubset)
        data = grow_cubic(N, int((N / (density * 1.2)) ** 0.333), method="linear")  # starting conformation
        PBC_width = (N / density) ** 0.333
        chains = [(N_chain * (k), N_chain * (k + 1), False) for k in range(num_chains)]  # now i
        reporter = HDF5Reporter(folder=simulation_save_folder, max_data_length=100)
        a = Simulation(N=N,
                        error_tol=0.01,
                        collision_rate=0.02,
                        integrator="variableLangevin",
                        platform="CUDA",
                        GPU="0",
                        PBCbox=False, # turn off bounding box
                        reporters=[reporter],
                        precision="mixed")  # platform="CPU", # GPU="1"
        
        a.set_data(data)  # initial polymer
        a.add_force(
            polychrom.forcekits.polymer_chains(
                a,
                chains=chains,
                nonbonded_force_func=polychrom.forces.heteropolymer_SSW,
                nonbonded_force_kwargs={
                    'attractionEnergy': 0,  # base attraction energy for all monomers
                    'attractionRadius': attraction_radius,
                    'interactionMatrix': interactionMatrix,
                    'monomerTypes': monomerTypes,
                    'extraHardParticlesIdxs': []
                },
                bond_force_kwargs={
                    'bondLength': 1,
                    'bondWiggleDistance': 0.05
                },
                angle_force_kwargs={
                    'k': angle_force
                },
                extra_bonds = extra_bond_pairs
            )
        )
        # ------------ initializing milker; adding bonds ---------
        kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
        bondDist = smcBondDist * a.length_scale
        activeParams = {"length": bondDist, "k": kbond}
        inactiveParams = {"length": bondDist, "k": 0}
        milker.setParams(activeParams, inactiveParams)
        milker.setup(bondForce=a.force_dict['harmonic_bonds'],
                        blocks=1)
        
        # If your simulation does not start, consider using energy minimization below
        a.local_energy_minimization()  # only do this at the beginning
        
        # this runs
        for i in range(restartSimulationEveryBlocks):  # loops over 100
            if i % saveEveryBlocks == (saveEveryBlocks - 1):
                a.do_block(steps=steps)
            else:
                a.integrator.step(steps)  # do steps without getting the positions from the GPU (faster)
                
        data = a.get_data()  # save data and step, and delete the simulation
        del a
        reporter.blocks_only = True  # Write output hdf5-files only for blocks
        time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)
        reporter.dump_data()

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
    