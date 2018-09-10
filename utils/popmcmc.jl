## Functions common to population MCMC algorithms

# Return log density for population of states
function eval_population_for( theta::Array{Float64,2}, calc_log_pdf::Function )
   popsize = size(theta,2)
   logpdf = Array{Float64}(popsize )
   for i in 1:popsize
      logpdf[i] = calc_log_pdf(theta[:,i])
   end
   return logpdf
end

#=
# WARNING: Does not get correct results!  (Why?)
function eval_population_parallel_for( theta::Array{Float64,2}, calc_log_pdf::Function )
   popsize = size(theta,2)
   ref = @parallel [calc_log_pdf(theta[:,i]) for i=1:popsize ]
   logpdf_darray = fetch(ref)
   return convert(Array{Float64,1},logpdf_darray)
end
=#

function eval_population_map( theta::Array{Float64,2}, calc_log_pdf::Function )
   map( i->calc_log_pdf(theta[:,i]), [1:size(theta,2)] )
   # nicer version, that requires knowing fancy syntax and doesn't parallelize as well
   #mapslices( calc_log_pdf, theta, 1 )
end

function eval_population_pmap( theta::Array{Float64,2}, calc_log_pdf::Function )
  pmap( i->calc_log_pdf(theta[:,i]), [1:size(theta,2)] )
  # Explicit version of how to implement your own pmap
  #refs = [ @spawn #calc_log_pdf(theta_init[:,i]) #for i in 1:size(theta_init,2) ]
  #map(fetch,  refs)
end

function eval_population_empty( theta::Array{Float64,2}, calc_log_pdf::Function )
   return zeros(size(theta,2))
end

eval_population( theta::Array{Float64,2}, calc_log_pdf::Function ) = eval_population_for( theta,calc_log_pdf )

#eval_population( theta::Array{Float64,2}, calc_log_pdf::Function ) = eval_population_parallel_for( theta,calc_log_pdf )

#eval_population( theta::Array{Float64,2}, calc_log_pdf::Function ) = eval_population_empty( theta,calc_log_pdf )


## Differential Evolution Sampler
#    based on ter Braak (2006) and Nelson et al. (2014)

# Generate ensemble of trial states according to DEMCMC
function generate_trial_states_demcmc( theta::Array{Float64,2}; gamma_o::Float64 = 2.38/sqrt(2*size(theta,1)), epsilon = 0.01  )
   num_param = size(theta,1)
   num_pop = size(theta,2)
   theta_trial = similar(theta)

   for i in 1:num_pop
      # Choose head (k) and tail (j) for perturbation vector
	  j = rand(1:(num_pop-1))
	  if( j >= i ) j = j+1 end
	  k = rand(1:(num_pop-2))
	  if( k >= i ) k = k+1 end
	  if( k >= j ) k = k+1 end

	  # Choose scale factor
	  scale = (1.0 + epsilon*randn()) * gamma_o

	  theta_trial[:,i] = theta[:,i] + scale * (theta[:,k] - theta[:,j])
   end
   return theta_trial
end

# Evolve population of states (theta_init) with target density
function run_demcmc( theta_init::Array{Float64,2}, calc_log_pdf::Function; num_gen::Integer = 100, epsilon::Float64 = 0.01 )
   @assert(num_gen >=1 )
   num_param = size(theta_init,1)
   num_pop = size(theta_init,2)
   @assert(num_pop > num_param)

   # Allocate arrays before starting loop
   pop = Array{Float64}( num_param, num_pop, num_gen)
   poplogpdf = Array{Float64}( num_pop, num_gen )
   accepts_chain = zeros(Int, num_pop )
   rejects_chain = zeros(Int, num_pop )
   accepts_generation = zeros(Int, num_gen )
   rejects_generation = zeros(Int, num_gen )

   # Make a working copy of the current population of parameter values
   theta = copy(theta_init)
   logpdf = eval_population( theta, calc_log_pdf )

   for g in 1:num_gen # first(pop.range):last(pop.range)  # loop over generations
      # Generate population of trial sets of model parameters
      gamma_o = (mod(g,10)!=0) ? 2.38/sqrt(2*num_param) : 1.0 # every 10th generation try full-size steps
      theta_trial = generate_trial_states_demcmc( theta, gamma_o=gamma_o, epsilon=epsilon )

      # Evaluate model for each set of trial parameters
	  logpdf_trial = eval_population( theta_trial, calc_log_pdf )

      # For each member of population
	  for i in 1:num_pop
		log_pdf_ratio = logpdf_trial[i] - logpdf[i]
	    if( (log_pdf_ratio>0) || (log_pdf_ratio>log(rand())) ) 	    # Decide whether to Accept
            theta[:,i] = theta_trial[:,i]
	    logpdf[i] = logpdf_trial[i]
            accepts_chain[i] = accepts_chain[i]+1
            accepts_generation[g] = accepts_generation[g]+1
        else
            rejects_chain[i] = rejects_chain[i]+1
            rejects_generation[g] = rejects_generation[g]+1
		end
	  end

      # Log results
	  if(true)  # (in(g,pop.range))
        pop[:,:,g] = copy(theta)
		poplogpdf[:,g] = copy(logpdf)
      end
   end
   return Dict("theta_last"=>theta, "logpdf_last"=>logpdf, "theta_all"=>pop, "logpdf_all"=>poplogpdf,
           "accepts_chain"=>accepts_chain, "rejects_chain"=>rejects_chain, "accepts_generation"=>accepts_generation, "rejects_generation"=>rejects_generation )
end


## Affine Invariant Sampler (based on Foreman-Macket et al. (2013) PASP 125, 306)

# Draw random variable from p(x) ~ x^(1/2) if x \in [1/a,a]
function draw_scale_affine_invariant( ; a::Float64 = 2.0 )
  u = rand()
  sqrta = sqrt(a)
  return (1.0/sqrta + u*(sqrta-1.0/sqrta))^2
end

# Generate ensemble of trial states according to affine invariant sampler
function generate_trial_states_affine_invariant( theta::Array{Float64,2}; a::Float64 = 2.0 )
   num_param = size(theta,1)
   num_pop = size(theta,2)
   @assert(num_param>=1)
   @assert(num_pop>num_param)

   theta_trial = similar(theta)
   z = Array{Float64}(num_pop)

   for i in 1:num_pop
      j = rand(1:(num_pop-1))
      if( j >= i ) j = j+1 end
      z[i] = draw_scale_affine_invariant( a=a)
      theta_trial[:,i] = theta[:,j] .+ z[i] .* (theta[:,i] - theta[:,j])
   end
   return (theta_trial,z)
end

function run_affine_pop_mcmc( theta_init::Array{Float64,2}, calc_log_pdf::Function;
   num_gen::Integer = 100, a::Float64 = 2.0 )
   @assert(num_gen >=1 )
   num_param = size(theta_init,1)
   num_pop = size(theta_init,2)
   @assert(num_pop > num_param)

   # Allocate arrays before starting loop
   pop = Array{Float64}(num_param, num_pop, num_gen)
   poplogpdf = Array{Float64}( num_pop, num_gen )
   accepts_chain = zeros(Int, num_pop )
   rejects_chain = zeros(Int, num_pop )
   accepts_generation = zeros(Int, num_gen )
   rejects_generation = zeros(Int, num_gen )

   # Make a working copy of the current population of parameter values
   theta = copy(theta_init)
   logpdf = eval_population( theta, calc_log_pdf )

   # Run multiple generations
   for g in 1:num_gen # first(pop.range):last(pop.range)  # loop over generations
      # Generate population of trial sets of model parameters
      (theta_trial, z_trial)  = generate_trial_states_affine_invariant( theta, a=a )

      # Evaluate model for each set of trial parameters
	  logpdf_trial= eval_population( theta_trial, calc_log_pdf )

      # For each member of population
	  for i in 1:num_pop
        log_pdf_ratio = logpdf_trial[i] - logpdf[i]
	    log_accept_prob = (num_param-1)*log(z_trial[i]) + log_pdf_ratio
        if( (log_accept_prob>0) || (log_accept_prob>log(rand())) ) 	    # Decide whether to Accept
            theta[:,i] = theta_trial[:,i]
            logpdf[i] = logpdf_trial[i]
            accepts_chain[i] = accepts_chain[i]+1
            accepts_generation[g] = accepts_generation[g]+1
        else
            rejects_chain[i] = rejects_chain[i]+1
            rejects_generation[g] = rejects_generation[g]+1
		end
	  end

      # Log results
	if(true)
        pop[:,:,g] = copy(theta)
	poplogpdf[:,g] = copy(logpdf)
      end
   end
   return Dict("theta_last"=>theta, "logpdf_last"=>logpdf, "theta_all"=>pop, "logpdf_all"=>poplogpdf,
           "accepts_chain"=>accepts_chain, "rejects_chain"=>rejects_chain, "accepts_generation"=>accepts_generation, "rejects_generation"=>rejects_generation )
end


## Population of Independent Random Walk Metropolic Hastings MCMC Samplers

# Generate ensemble of trial states
function generate_trial_states_indep( theta::Array{Float64,2}, covar_proposal::Array{Float64,2}  )
   num_param = size(theta,1)
   num_pop = size(theta,2)
   theta_trial = similar(theta)
   proposal_fact = cholfact(covar_proposal)
   for i in 1:num_pop
	  theta_trial[:,i] = theta[:,i] + proposal_fact[:L]*randn(num_param)
   end
   return theta_trial
end

# Estimate sample covariance matrix
function estimate_covariance(theta::Array{Float64,2})
  ndim = size(theta,1)
  ndata = size(theta,2)
  mu = [ mean(pop_init[i,:]) for i in 1:ndim ]
  sigma = Array{Float64}(ndim,ndim)
  for i in 1:ndim, j in 1:ndim
     sigma[i,j] = sum((theta[i,:].-mu[i]).*(theta[j,:].-mu[j]))
  end
  sigma /= ndata
end

# Evolve population of states (theta_init) with target density
function run_indep_rwmh_mcmc( theta_init::Array{Float64,2}, calc_log_pdf::Function, covar_prop::Array{Float64,2}; num_gen::Integer = 100 )
   @assert(num_gen >=1 )
   num_param = size(theta_init,1)
   num_pop = size(theta_init,2)
   @assert(num_pop > num_param)

   # Allocate arrays before starting loop
   pop = Array{Float64}( num_param, num_pop, num_gen)
   poplogpdf = Array{Float64}( num_pop, num_gen )
   accepts_chain = zeros(Int, num_pop )
   rejects_chain = zeros(Int, num_pop )
   accepts_generation = zeros(Int, num_gen )
   rejects_generation = zeros(Int, num_gen )

   # Make a working copy of the current population of parameter values
   theta = copy(theta_init)
   logpdf = eval_population( theta, calc_log_pdf )

   for g in 1:num_gen # first(pop.range):last(pop.range)  # loop over generations
      theta_trial = generate_trial_states_indep( theta, covar_prop )

      # Evaluate model for each set of trial parameters
	  logpdf_trial = eval_population( theta_trial, calc_log_pdf )

      # For each member of population
	  for i in 1:num_pop
		log_pdf_ratio = logpdf_trial[i] - logpdf[i]
	    if( (log_pdf_ratio>0) || (log_pdf_ratio>log(rand())) ) 	    # Decide whether to Accept
            theta[:,i] = theta_trial[:,i]
			logpdf[i] = logpdf_trial[i]
            accepts_chain[i] = accepts_chain[i]+1
            accepts_generation[g] = accepts_generation[g]+1
        else
            rejects_chain[i] = rejects_chain[i]+1
            rejects_generation[g] = rejects_generation[g]+1
		end
	  end

      # Log results
	  if(true)
        pop[:,:,g] = copy(theta)
		poplogpdf[:,g] = copy(logpdf)
      end
   end
   return Dict("theta_last"=>theta, "logpdf_last"=>logpdf, "theta_all"=>pop, "logpdf_all"=>poplogpdf,
           "accepts_chain"=>accepts_chain, "rejects_chain"=>rejects_chain, "accepts_generation"=>accepts_generation, "rejects_generation"=>rejects_generation )
end
