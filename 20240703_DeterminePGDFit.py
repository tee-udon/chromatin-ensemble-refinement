import gc
import os
import pickle
import subprocess
import time
import argparse
from joblib import Parallel, delayed
from utils import *
from functions import *

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

# Maybe I should pick one sample and then rerun structure optimization 10,000 times 
# dataset_sample is dataset_list[i]
def pgd(dataset_sample, param_dict):
    # Unpack parameters
    template_list = dataset_sample['template_chain_list']
    X = dataset_sample['observation_list']
    label_list = dataset_sample['label_list']
    
    # Generate flatten observation list
    observation_list = [squareform(pdist(x)).flatten() for x in X]
    
    # Generate random walk for guessing structures 
    num_monomers = param_dict['num_monomers']
    mean_bond_length = param_dict['mean_bond_length']
    std_bond_length = param_dict['std_bond_length'] 
    num_templates = param_dict['num_templates']
    measurement_error = param_dict['noise_std']
    num_probes = num_monomers
    num_candidates = num_templates
    
    candidate_polymer_list = [generate_gaussian_chain(num_monomers, mean_bond_length, std_bond_length) for _ in range(num_candidates)]
    candidate_flatten_list = [squareform(pdist(x)).flatten() for x in candidate_polymer_list]
    
    # Generate random number for guessing weights 
    candidate_weights = np.random.dirichlet(np.ones(num_candidates)) + jnp.finfo(jnp.float64).tiny # Add tiny value to avoid zero weights
    
    # Run projected gradient descent to determine the metastructures and their associated weights 
    # of the ensemble
    lp_structure = jit(lambda x: -log_posterior(x, observation_flatten_list, true_weights, measurement_error, num_probes, num_candidates, num_monomers**2))
    lp_weight = jit(lambda x: -log_posterior(candidate_flatten_list, observation_flatten_list, x, measurement_error, num_probes, num_candidates, num_monomers**2))
    
    pg = ProjectedGradient(fun=lp_structure, projection=projection_non_negative)
    pg_solution = pg.run(candidate_flatten_list)
    
    # Extract the predicted metastructures
    pred_means = pg_solution.params
    
    # Del pg and pg_solution to free up GPU memory 
    # and run garbage collection
    del pg
    del pg_solution
    torch.cuda.empty_cache()
    gc.collect()
    
    # Run projected gradient descent to determine the weights of the ensemble
    pg = ProjectedGradient(fun=lp_weight, projection=projection_simplex)
    pg_solution = pg.run(candidate_weights)
    
    # Extract the predicted weights 
    pred_weights = pg_solution.params
    
    # Del pg and pg_solution to free up GPU memory 
    # and run garbage collection
    del pg
    del pg_solution
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get true values and assign clusters to the metastructures and weights
    true_means = np.array([squareform(pdist(x)).flatten() for x in template_list])
    true_weights = np.array([np.sum(label_list == i) for i in range(len(template_list))]) / len(label_list)
    
    true_means_sorted, pred_means_sorted = assign_clusters(true_means, pred_means) 
    true_weights_sorted, pred_weights_sorted = assign_clusters(true_weights, pred_weights)
    
    
    
    