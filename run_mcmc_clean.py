## import necessary modules
import os, sys
import csv
from tqdm import tqdm
import numpy as np
import torch
# from scipy.special import logsumexp
import json

# check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import cmdstanpy
cmdstanpy.install_cmdstan()   # if you don't have cmdstan installed, uncomment this line
# cmdstanpy.rebuild_cmdstan()   # if cmdstan fails to compile, try uncommenting this line

from cmdstanpy import CmdStanModel
my_model = CmdStanModel(
    stan_file='./cryo-er.stan',
    cpp_options = {
        "STAN_THREADS": True,
    }
)

filename = sys.argv[1]

## detect number of CPU cores
import multiprocessing
n_cores = multiprocessing.cpu_count()
print(f"Number of CPU cores: {n_cores}")
parallel_chains = 4
threads_per_chain = int(n_cores / parallel_chains)
print(f"Number of threads per chain: {threads_per_chain}")

Dmat = None
if filename.endswith('.pt'):
    Dmat = torch.load(filename).to(torch.float64).numpy().T
elif filename.endswith('.npy'):
    Dmat = np.load(filename).T

output_directory = os.path.join(".", "output", filename.split("/")[-1].split(".")[0], "mcmc")
os.makedirs(output_directory, exist_ok = True)

## Write json files for reading into Stan program
N = Dmat.shape[0]
M = Dmat.shape[1]
print("Number of structures = %d, Number of images = %d." % (M, N))

## Read N_m, the number of conformations that are in the mth cluster
counts = np.ones(M, dtype=float) / M
log_Nm = np.log(counts)

json_filename = "%s/Dmat.json" % (output_directory)
stan_output_file = "%s/Stan_output" % (output_directory)
dictionary = {
    "M": M,
    "N": N,
    "logNm": list(log_Nm),
    "Dmat": [list(a) for a in Dmat],
}

json_object = json.dumps(dictionary, indent=4)

with open(json_filename, "w") as f:
    f.write(json_object)
    f.close()

## Run Stan model to perform MCMC sampling on posterior in Eq. 10 and 17
data_file = os.path.join(".", json_filename)
fit = my_model.sample(
    data=data_file,
    chains=4,
    sig_figs=8,
    parallel_chains=parallel_chains,
    threads_per_chain=threads_per_chain,
    iter_warmup=1000,
    iter_sampling=1000,
    show_console=True,
)

# Save Stan output, i.e., posterior samples, in CSV format, in a specified folder
fit.save_csvfiles(dir=stan_output_file)

print("Done!")