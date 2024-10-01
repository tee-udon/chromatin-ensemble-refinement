import os
import numpy as np
from scipy.special import logsumexp
import csv
import cryoER
from cryoER.analyze_mcmc import analyze_mcmc
import sys

import corner

output_directory = sys.argv[1]
N_center = int(sys.argv[2])
stan_directory = output_directory + '/Stan_output/'

nm = np.ones(N_center, dtype=float) / N_center
log_nm = np.log(nm)

log_weights = []
lp = []
files = sorted(os.listdir(stan_directory))[-4:]
print(files)

log_weights_d = []
for file in files:
    log_weights_chain = []
    lp_chain = []
    with open('%s/%s'%(stan_directory, file), newline='') as csvfile:
        reader = csv.DictReader(filter(lambda row: row[0]!='#', csvfile), )
        for row in reader:
            log_weights_row = [float(row["log_weights.%d"%i]) for i in range(1,N_center+1)]
            lp_chain.append(float(row["lp__"]))
            log_weights_chain.append(log_weights_row)
    log_weights = np.array(log_weights_chain)
    lp_chain = np.array(lp_chain)
    log_weights_d.append(log_weights)
    lp.append(lp_chain)

log_weights_d = np.array(log_weights_d)

from matplotlib import pyplot as plt
## make corner plot
log_weights_d_flat = log_weights_d.reshape(-1, N_center)
corner.corner(np.exp(log_weights_d_flat), labels=[str(i) for i in range(N_center)])
plt.savefig(output_directory + 'corner_plot.png')

log_factor = log_weights_d - logsumexp(log_weights_d, axis=2)[:,:,None]

factor = np.exp(log_factor)         # sampled reweighting factor
factor_mean = factor.mean((0,1))
factor_std = factor.std((0,1))

factor_mean_std = np.vstack((factor_mean, factor_std)).T
np.savetxt(output_directory + "reweighting_factor.txt", factor_mean_std, fmt='%.6f')

log_rewtprob = log_weights_d + log_nm[None,None,:]
log_rewtprob -= logsumexp(log_rewtprob, axis=2)[:,:,None]

rewtprob = np.exp(log_rewtprob)     # reweighted probability
rewtprob_mean = rewtprob.mean((0,1))
rewtprob_std = rewtprob.std((0,1))

rewtprob_mean_std = np.vstack((rewtprob_mean, rewtprob_std)).T
np.savetxt(output_directory+"reweighted_prob.txt", rewtprob_mean_std, fmt='%.6f')

lp = np.array(lp)
np.savetxt(output_directory+"lp.txt", lp, fmt='%.6f')
