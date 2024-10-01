import sys
sys.path.append(r"/mnt/ceph/users/tudomlumleart/00_VirtualEnvironments/teeu/lib/python3.10/site-packages")
sys.path.append(r"/mnt/home/tudomlumleart/.local/lib/python3.10/site-packages/")
sys.path.append(r"/mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jupyter-gpu/lib/python3.10/site-packages")
from utils import *
from functions import *
from sklearn.mixture import BayesianGaussianMixture
import sklearn

import os
import scipy.io

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

if __name__ == '__main__':
    # Load dataset from file
    print('Loading dataset...')
    folder_path = '/mnt/home/tudomlumleart/ceph/05_Sox9Dataset/'

    # List all .mat files in the folder and load them
    cnc_maps = scipy.io.loadmat(folder_path + 'cncMaps.mat')['cncMaps']
    esc_maps = scipy.io.loadmat(folder_path + 'escMaps.mat')['escMaps']
    
    # Load polys data and then perform linear interpolation
    # List all .mat files in the folder and load them
    cnc_polys = scipy.io.loadmat(folder_path + 'cncPols.mat')['cncPols']
    esc_polys = scipy.io.loadmat(folder_path + 'escPols.mat')['escPols']
    
    esc_polys_interp = interpolate_polymers(esc_polys)[:80, :, :]
    cnc_polys_interp = interpolate_polymers(cnc_polys)[:80, :, :]
    
    esc_maps_interp = np.array([squareform(pdist(esc_polys_interp[:, :, i])) for i in range(esc_polys_interp.shape[2])])
    cnc_maps_interp = np.array([squareform(pdist(cnc_polys_interp[:, :, i])) for i in range(cnc_polys_interp.shape[2])])
    
    n_comp = 250

    # print('Fitting Bayesian Gaussian Mixture Models...')
    # print('ESC')
    # # Create and fit the Bayesian Gaussian Mixture Model
    # bgmm_esc = BayesianGaussianMixture(
    #     n_components=n_comp,      # Maximum number of components
    #     covariance_type='full', # Type of covariance parameters
    #     max_iter=1000,         # Maximum number of iterations  
    #     verbose=2
    # )
    
    esc_maps_interp_flat = np.array([x.flatten() for x in esc_maps_interp])
    # bgmm_esc.fit(esc_maps_interp_flat)
    
    # print('CNC')
    # # Create and fit the Bayesian Gaussian Mixture Model
    # bgmm_cnc = BayesianGaussianMixture(
    #     n_components=n_comp,      # Maximum number of components
    #     covariance_type='full', # Type of covariance parameters
    #     max_iter=1000,         # Maximum number of iterations  
    #     verbose=2
    # )

    cnc_maps_interp_flat = np.array([x.flatten() for x in cnc_maps_interp])
    # bgmm_cnc.fit(cnc_maps_interp_flat)
    
    print('All')
    bgmm_all = BayesianGaussianMixture(
        n_components=n_comp,      # Maximum number of components
        covariance_type='full', # Type of covariance parameters
        max_iter=1000,         # Maximum number of iterations  
        verbose=2
    )
    
    all_maps_interp = np.concatenate([esc_maps_interp_flat, cnc_maps_interp_flat], axis=0)
    
    all_maps_interp_flat = np.array([x.flatten() for x in all_maps_interp])
    bgmm_all.fit(all_maps_interp_flat)
    
    # Write the models to pickle files
    pickle_dict = {
        # 'bgmm_esc': bgmm_esc,
        # 'bgmm_cnc': bgmm_cnc,
        'bgmm_all': bgmm_all
    }
    
    # Dump the pickle dict to a file
    print('Writing models to file...')
    with open(folder_path + 'bgmm_models_250.pkl', 'wb') as f:
        pickle.dump(pickle_dict, f)