from _utils import *

def reweight_samples(
    distance_map_list,
    distance_map_flat_list,
    sample_labels,
    num_microstates,
    save_dir,
    method='PCA',
    slurm_file=None):
    # Add docstring 
    """
    """
    num_probes = distance_map_list.shape[1]
    sample_labels = np.array(sample_labels)
    
    if slurm_file is None:
        slurm_file = '/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/slurm/2024_RunPythonScript.sh'        
    
    print('PCA Fitting...')
    if method == 'PCA':
        pca = PCA(n_components=2)
        pca.fit(distance_map_flat_list)
        pca_samples = []
        unique_labels = np.unique(sample_labels)
        for label in unique_labels:
            pca_samples.append(pca.transform(distance_map_flat_list[sample_labels == label, :]))
            
        df_sample_list = []
        for i, label in enumerate(unique_labels):
            df_sample = pd.DataFrame(pca_samples[i], columns=['PC1', 'PC2'])
            df_sample['label'] = label
            df_sample_list.append(df_sample)
        df_samples = pd.concat(df_sample_list, axis=0)
        
        min_pc1 = df_samples['PC1'].min()
        max_pc1 = df_samples['PC1'].max()
        min_pc2 = df_samples['PC2'].min()
        max_pc2 = df_samples['PC2'].max()
        
        num_microstate_per_axis = np.round(np.sqrt(num_microstates), 0).astype(int)
        
        microstate_distance_maps = generate_microstates(
            min_pc1, max_pc1, min_pc2, max_pc2, num_microstate_per_axis, pca)
       
    print('Calculating likelihood...') 
    microstate_distance_maps_jnp = jnp.array(microstate_distance_maps)
    print(microstate_distance_maps_jnp.shape)
    sample_std = []
    sample_ll = []
    sample_num = []
    for label in unique_labels:
        curr_condition = jnp.array(distance_map_list[sample_labels == label, :, :])
        print(curr_condition.shape)
        curr_std = batch_calculate_variances(curr_condition,
                                             microstate_distance_maps_jnp,
                                             num_probes) ** 0.5
        sample_std.append(curr_std)
        
        curr_ll = []
        for y in tqdm(curr_condition):
            curr_ll.append(compute_loglikelihood_for_y(
                y.flatten(), microstate_distance_maps_jnp, 
                curr_std, num_probes).tolist()) 
        
        sample_num.append(curr_condition.shape[0])
        sample_ll.append(curr_ll)

    lpm = [(logprior(x, num_probes)).tolist() for x in microstate_distance_maps]
    
    # Load stan model 
    my_model = CmdStanModel(
        stan_file='/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/stan/20240715_WeightOptimization.stan',
        cpp_options = {
            "STAN_THREADS": True,
        }
        )
    
    print('Saving data...')
    for i, label in tqdm(enumerate(unique_labels)):
        output_dir = os.path.join(save_dir, label)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        json_filename = os.path.join(output_dir, 'data.json')
        stan_output_file = os.path.join(output_dir, 'stan_output')
        
        data_dict = {
            'N': sample_num[i],
            'M': num_microstates,
            'll_map': sample_ll[i],
            'lpm_vec': lpm
        }
        
        json_obj = json.dumps(data_dict, indent=4)
        
        with open(json_filename, 'w') as f:
            f.write(json_obj)
            f.close()
    
    print('Submitting slurm jobs...')       
    submit_mcmc_slurm(save_dir, slurm_file)
    