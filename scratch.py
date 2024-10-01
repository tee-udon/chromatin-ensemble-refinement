def evaluate_kl_div(mcmc_result_folder):
    stan_directory = mcmc_result_folder
    
    log_weights = []
    lp = []
    files = sorted(os.listdir(stan_directory))[-4:]
    
    num_metastructures = extract_specific_number(stan_directory, 3)
    weight_dist = extract_specific_number(stan_directory, 4)
    noise_level = extract_specific_number(stan_directory, 5)
    
    log_weights_d = []
    for file in files:
        log_weights_chain = []
        lp_chain = []
        with open('%s/%s'%(stan_directory, file), newline='') as csvfile:
            reader = csv.DictReader(filter(lambda row: row[0]!='#', csvfile), )
            for row in reader:
                log_weights_row = [float(row["log_weights.%d"%i]) for i in range(1,num_metastructures+1)]
                lp_chain.append(float(row["lp__"]))
                log_weights_chain.append(log_weights_row)
        log_weights = np.array(log_weights_chain)
        lp_chain = np.array(lp_chain)
        log_weights_d.append(log_weights)
        lp.append(lp_chain)
    log_weights_d = np.array(log_weights_d)
    
    log_weights_d = np.array(log_weights_d)
    log_weights_d_flat = log_weights_d.reshape(-1, num_metastructures)
    
    if num_metastructures <= 10:
        corner.corner(np.exp(log_weights_d_flat), labels=[str(i) for i in range(num_metastructures)])
        plt.savefig(stan_directory + 'corner_plot.png')
        
    pred_weights = np.exp(log_weights_d_flat)
    true_weights_dir =  os.path.join(os.path.split(stan_directory)[0], 'true_weights.txt')
    true_weights = np.loadtxt(true_weights_dir, delimiter=' ')
    
    kl_div_list = [scipy.special.kl_div(true_weights, x).sum() for x in pred_weights]
    
    # generate a pandas dataframe that contain kl_div_list 
    # other 3 columns are num_metastructures, weight_dist, noise_level
    
    
    return kl_div_list, num_metastructures, weight_dist, noise_level
