# This version of MCMC is compatible with the revised version of 
# metastructure prior probability calculation and also new function names

# This import statement should import all necessary functions and packages 
from _utils import *

def run_mcmc(save_dir):
    # Load observations 
    # Load dataset from file
    folder_path = '/mnt/home/tudomlumleart/ceph/05_Sox9Dataset/'
    
    num_monomers = 80
    
    # Load polys data and then perform linear interpolation
    # List all .mat files in the folder and load them
    cnc_polys = scipy.io.loadmat(folder_path + 'cncPols.mat')['cncPols'][:num_monomers, :, :]
    esc_polys = scipy.io.loadmat(folder_path + 'escPols.mat')['escPols'][:num_monomers, :, :]
    
    esc_polys_interp = interpolate_polymers(esc_polys)
    cnc_polys_interp = interpolate_polymers(cnc_polys)
    
    esc_maps_interp = np.array([squareform(pdist(esc_polys_interp[:80, :, i])) for i in range(esc_polys_interp.shape[2])])
    cnc_maps_interp = np.array([squareform(pdist(cnc_polys_interp[:80, :, i])) for i in range(cnc_polys_interp.shape[2])])
    esc_maps_interp_flat = np.array([x.flatten() for x in esc_maps_interp])
    cnc_maps_interp_flat = np.array([x.flatten() for x in cnc_maps_interp])
    all_maps_interp = np.concatenate((esc_maps_interp, cnc_maps_interp), axis=0)
    all_maps_interp_flat = np.concatenate((esc_maps_interp_flat, cnc_maps_interp_flat), axis=0)
    
    pca = PCA(n_components=2)
    pca.fit(all_maps_interp_flat)
    esc_maps_pca = pca.transform(esc_maps_interp_flat)
    cnc_maps_pca = pca.transform(cnc_maps_interp_flat)
    
    esc_df = pd.DataFrame(esc_maps_pca, columns=['PC1', 'PC2'])
    esc_df['label'] = 'ESC'
    cnc_df = pd.DataFrame(cnc_maps_pca, columns=['PC1', 'PC2'])
    cnc_df['label'] = 'CNC'
    all_df = pd.concat([esc_df, cnc_df], axis=0)
    
    # Find lower bound and upper bound of PC1 and PC2 data
    l = 0.01
    u = 1-l

    pc1_l = all_df['PC1'].quantile(l)
    pc1_u = all_df['PC1'].quantile(u)
    pc2_l = all_df['PC2'].quantile(l)
    pc2_u = all_df['PC2'].quantile(u)

    
    pc1_grid = np.linspace(pc1_l, pc1_u, 50)
    pc2_grid = np.linspace(pc2_l, pc2_u, 50)
    
    # Generate combination of pc1 and pc2 values
    pc1_grid, pc2_grid = np.meshgrid(pc1_grid, pc2_grid)
    
    # put this into a dataframe
    pc1_grid_flat = pc1_grid.flatten()
    pc2_grid_flat = pc2_grid.flatten()
    pc1_pc2_df = pd.DataFrame({'PC1': pc1_grid_flat, 'PC2': pc2_grid_flat})
    pc1_pc2_df['label'] = 'metastructures'
    
    # Sort PC2 in descending order while keeping PC1 in ascending order
    pc1_pc2_df = pc1_pc2_df.sort_values(by=['PC1', 'PC2'], ascending=[True, False], ignore_index=True)  

    metastr_from_pca = pca.inverse_transform(pc1_pc2_df[['PC1', 'PC2']])   

    templates_flatten = metastr_from_pca
    
    measurement_error_esc = [calculate_conformational_variance(esc_maps_interp, x.reshape(80, 80))**0.5 for x in templates_flatten]
    measurement_error_cnc = [calculate_conformational_variance(cnc_maps_interp, x.reshape(80, 80))**0.5 for x in templates_flatten]
    
    # Generate log prior for metastructures 
    lpm = [(logprior(x, num_monomers)).tolist() for x in templates_flatten]
    
    # Generate log likelihood for observations given metastructures 
    ll_esc = [[(loglikelihood(y, x, z, num_monomers)).tolist() for x, z in zip(templates_flatten, measurement_error_esc)] for y in esc_maps_interp_flat]
    ll_cnc = [[(loglikelihood(y, x, z, num_monomers)).tolist() for x, z in zip(templates_flatten, measurement_error_cnc)] for y in cnc_maps_interp_flat]
    
    N_cnc = cnc_maps_interp_flat.shape[0]
    N_esc = esc_maps_interp_flat.shape[0]
    M = templates_flatten.shape[0]
    
    print('Finished preprcoessing data')
    
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
    
    conditions = ['ESC', 'CNC']
    for condition in conditions:
        output_dir = os.path.join(save_dir, condition)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Write json files for reading into stan program
        json_filename = os.path.join(output_dir, 'data.json')
        stan_output_file = os.path.join(output_dir, 'stan_output')
        
        if condition == 'ESC':
            data_dict = {
                "M": M,
                "N": N_esc,
                "ll_map": ll_esc,
                "lpm_vec": lpm,
            }
        elif condition == 'CNC': 
            data_dict = {
                "M": M,
                "N": N_cnc,
                "ll_map": ll_cnc,
                "lpm_vec": lpm,
            }
            
        json_obj = json.dumps(data_dict, indent=4)

        with open(json_filename, 'w') as json_file:
            json_file.write(json_obj)
            json_file.close()
            
        # Run Stan model to perform MCMC sampling
        data_file = json_filename
        
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
        
        
    
# Run file from the command line 
if __name__ == '__main__':
    save_dir = '/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/MCMC_results/20240905_RunWeightMCMC_Sox9_PCA_utilsV2.py/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
            
    print(f"Running MCMC ...")
    run_mcmc(save_dir)
    
