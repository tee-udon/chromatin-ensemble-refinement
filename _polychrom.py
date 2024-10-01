import sys
# add path to my polychrom installation 
sys.path.append(r"/mnt/ceph/users/tudomlumleart/00_VirtualEnvironments/teeu/lib/python3.10/site-packages")
sys.path.append(r"/mnt/home/tudomlumleart/.local/lib/python3.10/site-packages/")
sys.path.append(r"/mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jupyter-gpu/lib/python3.10/site-packages")
sys.path.append(r"/mnt/home/tudomlumleart/.local/lib/python3.10/site-packages/")

import os 
import h5py
import ast 
import math
import copy
import csv 
# import jax 
# import jaxopt
import scipy
# import mpltern
import shutil
import multiprocessing
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
import extrusion1Dv2 as ex1D
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

from LEBondUpdater import bondUpdater

import polychrom
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file
from polychrom.simulation import Simulation
from polychrom import polymerutils
from polychrom import forces
from polychrom import forcekits


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