import sys

sys.path.append(r"/mnt/ceph/users/tudomlumleart/00_VirtualEnvironments/teeu/lib/python3.10/site-packages")
sys.path.append(r"/mnt/home/tudomlumleart/.local/lib/python3.10/site-packages/")
sys.path.append(r"/mnt/home/tudomlumleart/ceph/00_VirtualEnvironments/jupyter-gpu/lib/python3.10/site-packages")

import os
import pickle
import subprocess
import time
import argparse
from joblib import Parallel, delayed
from utils import *

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

def fit_bgmm(observation_list, n_comp=120):
    """
    Fits a Bayesian Gaussian Mixture Model (BGMM) to the given observation list.

    Parameters
    ----------
    observation_list : array-like
        The list of observations to fit the BGMM to.
    n_comp : int, optional
        The maximum number of components in the BGMM. Default is 120.

    Returns
    -------
    bgmm : BayesianGaussianMixture
        The fitted BGMM model.
    """

    bgmm = BayesianGaussianMixture(
        n_components=n_comp,      # Maximum number of components
        covariance_type='diag', # Type of covariance parameters
        max_iter=1000,         # Maximum number of iterations
        init_params='k-means++', # Method of initialization
        n_init=1
    )

    bgmm.fit(observation_list)
    
    return bgmm


# Define a function to evaluate the performance of the BGMM model
def evaluate_bgmm(dataset_list):
    """
    Evaluate the performance of the Bayesian Gaussian Mixture Model (BGMM) on a given dataset.

    Parameters:
    - pickle_file (str): The path to the pickle file containing the dataset.

    Returns:
    - performance_dict (dict): A dictionary containing the performance metrics of the BGMM model.
        - mean_mse (float): The mean squared error between the true and predicted means.
        - cov_mse (float): The mean squared error between the true and predicted covariances.
        - cov_corr (float): The correlation between the true and predicted covariances.
        - weight_corr (float): The correlation between the true and predicted weights.
        - weight_kl (float): The KL divergence between the true and predicted weights.
        - log_likelihood (float): The log-likelihood of the model.
        - ami_score (float): The adjusted mutual information score of the model.
    """
    template_list = dataset_list['template_chain_list']
    X = dataset_list['observation_list']
    label_list = dataset_list['labels']
    
    observation_list = [squareform(pdist(x)).flatten() for x in X]
    
    # Fit the BGMM model
    fitted_bgmm = fit_bgmm(observation_list, n_comp=len(template_list))
    
    # Make sure that all input lists are numpy arrays
    template_list = np.array(template_list)
    observation_list = np.array(observation_list)
    label_list = np.array(label_list)
    
    # Extract the means, covariances, and weights from the fitted BGMM model
    pred_means = fitted_bgmm.means_
    pred_covs = fitted_bgmm.covariances_
    pred_weights = fitted_bgmm.weights_
    
    # Assign the clusters based on the minimum distance between the true and predicted means
    true_means = np.array([squareform(pdist(x)).flatten() for x in template_list])
    true_covs = np.array([np.var(observation_list[label_list == i], axis=0) for i in np.unique(label_list)])
    true_weights = np.array([np.sum(label_list == i) for i in np.unique(label_list)]) / len(label_list)
    true_idx, pred_idx = assign_clusters(true_means, pred_means, True)
    true_structure, pred_structure = assign_clusters(true_means, pred_means, False)
    
    # Calculate the mean squared error between the true and predicted means
    mean_mse = np.mean([np.linalg.norm(x-y) for x, y in zip(true_structure, pred_structure)])
    
    # Calculate the mean squared error between the true and predicted covariances
    cov_mse = MSD_covariances(true_covs[true_idx], pred_covs[pred_idx])
    
    # Calculate the correlation between the true and predicted covariances
    cov_corr = corr_covariances(true_covs[true_idx], pred_covs[pred_idx])
    
    # Calculate the correlation between the true and predicted weights
    weight_corr = corr_weights(true_weights[true_idx], pred_weights[pred_idx])
    
    # Calculate the KL divergence between the true and predicted weights
    weight_kl = np.sum(scipy.special.kl_div(true_weights.flatten(), pred_weights.flatten()))
    
    # Calculate the log-likelihood of the model
    log_likelihood = fitted_bgmm.score(observation_list)
    
    # Calculate the AMI score of the model
    ami_score = adjusted_mutual_info_score(label_list, fitted_bgmm.predict(observation_list))
    
    # Save the performance metrics in a dictionary
    performance_dict = {
        'mean_mse': mean_mse,
        'cov_mse': cov_mse,
        'cov_corr': cov_corr,
        'weight_corr': weight_corr,
        'weight_kl': weight_kl,
        'log_likelihood': log_likelihood,
        'ami_score': ami_score
    }
    
    # return the performance metrics
    return performance_dict

def write_performance(performance_list, param_dict, pickle_file):
    # Generate a new folder to save the results
    # Check if the folder already exists
    split_path = pickle_file.split('/')
    pickle_fname = split_path[-1]
    result_dir = '/'.join(split_path[:-1]) + '/results_BGMMFit_20240730/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    
    # Write each dictionary in performance list to a csv file 
    # Columns names include the parameters used to generate the dataset
    # and the performance metrics
    # The file name should be the same as the pickle file with .txt extension  
    fname = result_dir + pickle_fname.replace('.pkl', '.txt')
    with open(fname, 'w') as f:
        writer = csv.writer(f)
        performance_dict = performance_list[0]
        writer.writerow(list(param_dict.keys()) + list(performance_dict.keys()))
        for performance_dict in performance_list:
            writer.writerow(list(param_dict.values()) + list(performance_dict.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process pickle data.')
    parser.add_argument('pickle_data', type=str, help='Pickle data to process.')

    args = parser.parse_args()
    dataset_list, param_dict = load_dataset(args.pickle_data)
    performance_list = Parallel(n_jobs=-1)(delayed(evaluate_bgmm)(dataset) for dataset in dataset_list)
    write_performance(performance_list, param_dict, args.pickle_data)
    
    