from _utils import *

# write a function that runs MCMC on a data in the directory
def run_mcmc(mcmc_dir):
    # Load stan model 
    my_model = CmdStanModel(
        stan_file='/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/stan/20240715_WeightOptimization.stan',
        cpp_options = {
            "STAN_THREADS": True,
        }
    )
    
    n_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {n_cores}")
    parallel_chains = 4
    threads_per_chain = int(n_cores / parallel_chains)
    print(f"Number of threads per chain: {threads_per_chain}")
    
    json_filename = os.path.join(mcmc_dir, 'data.json')
    stan_output_dir = os.path.join(mcmc_dir, 'stan_output')
    
    fit = my_model.sample(
        data=json_filename,
        chains=4,
        sig_figs=8,
        parallel_chains=parallel_chains,
        threads_per_chain=threads_per_chain,
        iter_warmup=1000,
        iter_sampling=1000,
        show_console=True,
    )
    
    # Save Stan output, i.e., posterior samples, in CSV format, in a specified folder
    fit.save_csvfiles(dir=stan_output_dir)
    
if __name__ == '__main__':
    # first argument is the directory containing the data
    mcmc_dir = sys.argv[1]
    run_mcmc(mcmc_dir)