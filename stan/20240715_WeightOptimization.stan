functions {
    real log_posterior(
        array[,] real ll_slice,
        int start,
        int end,
        vector lpm_vec,
        vector lpw_vec,
        vector log_weights 
        ) {
            real total = 0;
            for (i in 1:end-start+1) { // for each observation (row)
                total += log_sum_exp(to_vector(ll_slice[i]) + lpm_vec + lpw_vec + log_weights);
            }

            return total;
        }
        
}

data {
    int<lower=0> M; // number of metastructures M
    int<lower=0> N; // number of observations N
    array[N, M] real ll_map; // log-likelihood map of all N observations across M metastructures
    vector[M] lpm_vec; // log-prior for M metastructures
}

transformed data{
    int grain_size = 1; // calculate log-posterior by adding the contribution from each observation
    vector[M] lpw_vec = rep_vector(1.0/M, M); // log-prior for each weight (uniform distribution)
    vector[M] ones = rep_vector(1, M); // hyperparameters for dirichlet distribution
}

parameters {
    simplex[M] weights;
}

transformed parameters {
    vector[M] log_weights = log(weights);
}

model {
    weights ~ dirichlet(ones);
    target += reduce_sum(log_posterior, ll_map, grain_size, lpm_vec, lpw_vec, log_weights);
}