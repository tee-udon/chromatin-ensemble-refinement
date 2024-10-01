functions {
    real log_posterior(
        array[,] real ll_slice,
        int start,
        int end,
        vector lpw_vec,
        vector log_weights 
        ) {
            real total = 0;
            for (i in 1:end-start+1) { // for each observation (row)
                total += log_sum_exp(to_vector(ll_slice[i]) + lpw_vec + log_weights);
            }

            return total;
        }
        
    real log_likelihood(
        vector obs_dmap_flat,
        vector meta_dmap_flat,
        real noise_std,
        int N_probe
    ) {
        real frob_norm;
        real prefactor;
        real gaussian;
        real ll;  

        frob_norm = sqrt(squared_distance(obs_dmap_flat, meta_dmap_flat));
        prefactor = -N_probe^2 * log(sqrt(2 * pi()) * noise_std);
        gaussian = -frob_norm / (2 * noise_std^2);
        ll = prefactor + gaussian;

        return ll;
    }

    vector extract_diagonal(vector flat_matrix, int N) {
        vector[N-1] diagonal_elements;
        
        for (i in 1:N-1) {
        diagonal_elements[i] = flat_matrix[(i-1) * N + (i+1)];
        }
        
        return diagonal_elements;
    }

    real log_prior_meta(
        vector meta_dmap_flat,
        int N_probe
    ) {
        vector[N_probe-1] b = extract_diagonal(meta_dmap_flat, N_probe);
        real R_sq = (meta_dmap_flat[N_probe])^2;
        real scaling_factor;
        real gaussian;
        real lpm;

        scaling_factor = 1.5 * log(3/(2 * pi() * N_probe * (mean(b))^2));
        gaussian = -3 * R_sq / (2 * N_probe * (mean(b))^2);
        lpm = scaling_factor + gaussian;

        return lpm;
    }
}

data {
    int<lower=0> M; // number of metastructures M
    int<lower=0> N; // number of observations N
    int<lower=0> N_probe; // number of probes
    real noise_std; // noise std 
    matrix[N, N_probe * N_probe] dmap; // 2D distance map of all N observations 
}

transformed data{
    int grain_size = 1; // calculate log-posterior by adding the contribution from each observation
    vector[M] lpw_vec = rep_vector(1.0/M, M); // log-prior for each weight (uniform distribution)
    vector[M] ones = rep_vector(1, M); // hyperparameters for dirichlet distribution
}

parameters {
    simplex[M] weights;
    matrix<lower=0>[M, N_probe * N_probe] mu;
}

transformed parameters {
    vector[M] log_weights = log(weights);
    matrix[N, M] ll; // log likelihood
    for (n in 1:N) {
        for (m in 1:M) {
            ll[n, m] = log_likelihood(to_vector(dmap[n]), to_vector(mu[m]), noise_std, N_probe);
        }
    }
    vector[M] lpm; // log prior metastructures
    for (m in 1:M) {
        lpm[m] = log_prior_meta(to_vector(mu[m]), N_probe);
    }
    for (n in 1:N){
        ll[n] = to_row_vector(log(exp(to_vector(ll[n])) + exp(lpm)));
    }
}

model {
    weights ~ dirichlet(ones);
    for (m in 1:M) {
        for (n in 1:N_probe * N_probe) {
            mu[m, n] ~ exponential(0.5); // assuming normal prior on metastructure mu
        }
            
    }
    
    target += reduce_sum(log_posterior, to_array_2d(ll), grain_size, lpw_vec, log_weights);
}