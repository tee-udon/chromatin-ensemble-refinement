if __name__ == '__main__':
    from utils import *

    params_dict = {
        'num_monomers': [100],
        'mean_bond_length': [10],
        'std_bond_length': [20],
        'num_templates': [2, 5, 10, 50, 100, 250, 500],
        'weights_dist': [0, 1, 1000],
        'gaussian_noise_std': np.array([0.1, 0.5, 1, 2, 5]) * 20,
        'num_observations': [10000],
        'num_datasets': [1000],
        'save_dir': ['/mnt/home/tudomlumleart/ceph/03_GaussianChainSimulation/20240627/']
    }

    generate_dataset(params_dict)