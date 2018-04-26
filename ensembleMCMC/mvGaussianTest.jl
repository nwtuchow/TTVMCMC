#multivariate gaussian test of ensemble samplers
# Create a function that returns target density to sample from (initially a multivariable normal)
ndim = 2    # number of model parameters
srand(157)  # seed the random number generator
tmp = rand(ndim,ndim);
target_covar = tmp'*tmp
target_covar_fact = cholfact(target_covar)


function log_target_pdf(theta::Array{Float64,1}, beta::Float64 = 1.0)
  @assert(size(theta,1)== size(target_covar_fact,1) == size(target_covar_fact,2) )
  ll = -0.5*( dot(vec(theta),vec(target_covar_fact\theta)) +  2*logdet(target_covar_fact) + size(theta,1)*log(2pi) )
  return beta*ll
end


# Create an initial population of parameter values
popsize = max(16,floor(Int64, 4*ndim))
offset = 0.0
scale = 1.0
pop_init = Array(Float64,ndim,popsize);
for i in 1:popsize
    pop_init[:,i] = offset .+ scale.* (target_covar_fact[:L] * randn(ndim))
end


# Pause to plot initial population
using PyPlot

plot(vec(pop_init[1,:]),vec(pop_init[2,:]),"r.")
xlabel(L"\theta_1")
ylabel(L"\theta_2")


# Load functions to perform population MCMC from a file
include("popmcmc.jl")


# Setup two possible proposal densities for use with Independent Random Walk Metropolis Hasting (IRMMH) MCMC.
rwmh_prop_covar = 2.38^2/size(pop_init,1)*estimate_covariance(pop_init)
rwmh_prop_covar_diag = copy(rwmh_prop_covar)
for i in 1:ndim, j in 1:ndim
   if i!=j
     rwmh_prop_covar_diag[i,j] = 0.0
   end
end
rwmh_prop_covar_diag

#=1a.  Assess the sampling efficiency of Independent Random Walk Metropolis-Hasting MCMC with a diagonal Proposal Density
Run IRMMH MCMC simulations with a diagonal proposal density and inspect plots showing the
evolution of the model parameters in the first few chains, using the code below.=#

results_rwmh = run_indep_rwmh_mcmc(pop_init,  log_target_pdf, rwmh_prop_covar_diag, num_gen= 500)

plot_trace(results_rwmh,1)

plot_trace(results_rwmh,2)

rwmh_prop_covar_try = 10.*rwmh_prop_covar_diag
results_rwmh = run_indep_rwmh_mcmc(pop_init,  log_target_pdf, rwmh_prop_covar_try, num_gen= 500);
plot_trace(results_rwmh,1)
plot_trace(results_rwmh,2)
# and
rwmh_prop_covar_try = 0.01*rwmh_prop_covar_diag
# followed by same code from above to run and plot
results_rwmh = run_indep_rwmh_mcmc(pop_init,  log_target_pdf, rwmh_prop_covar_try, num_gen= 500);
plot_trace(results_rwmh,1)
plot_trace(results_rwmh,2)

#=1b.  Assess the sampling efficiency of Independent Random Walk Metropolis-Hasting MCMC with a non-diagonal Proposal Density
Next, run IRWMH MCMC simulations with a non-diagonal proposal density and inspect the results visually.=#
results_rwmh = run_indep_rwmh_mcmc(pop_init,  log_target_pdf, rwmh_prop_covar, num_gen= 500);
plot_trace(results_rwmh,1)
plot_trace(results_rwmh,2)

rwmh_prop_covar_try = 2.0*rwmh_prop_covar
results_rwmh = run_indep_rwmh_mcmc(pop_init,  log_target_pdf, rwmh_prop_covar_try, num_gen= 500);
plot_trace(results_rwmh,1)
plot_trace(results_rwmh,2)

#=1c.  Assess the sampling efficiency of Differential Evolution MCMC
Run a Differential Evolution MCMC (DE-MCMC) simulation and visually inspect the evolution of the model parameters.=#
results_demcmc = run_demcmc( pop_init,  log_target_pdf, num_gen= 500);
plot_trace(results_demcmc,1)
plot_trace(results_demcmc,2)

#=1d. Assess the sampling efficiency of an Affine Invariant MCMC
Run an Affine Invariant Population MCMC (AIP-MCMC) simulation and visually inspect the evolution of the model parameters.=#
results_affine = run_affine_pop_mcmc(pop_init,  log_target_pdf, num_gen= 500); results_affine["theta_last"]
plot_trace(results_affine,1)
plot_trace(results_affine,2)

#=2.  Assess Burn-in Time Required with Population MCMC Algorithm
For a real problem, we wouldn’t be able to start with a population drawn from the target density.  Therefore, we’d need to let the Markov chain run “burn-in” for enough iterations that the initial conditions no longer have a significant effect on the samples being generated.  In this exercise, you are to compare the three sampling algorithms again, but choosing an initial population of parameter values that is offset from the mode of the target density and/or changed in size or shape.  For example, you might try an initial population like: =#
popsize = max(16,floor(Int64, 4*ndim))
offset = [10.0, 10.0]
scale = 0.01
pop_init_poor = Array(Float64,ndim,popsize);
for i in 1:popsize
    pop_init_poor[:,i] = offset .+ scale*randn(ndim)
end

#Then rerun the three algorithms, each starting from that initial population:
rwmh_prop_covar = 2.38^2/size(pop_init,1)*estimate_covariance(pop_init_poor)
results_rwmh = run_indep_rwmh_mcmc(pop_init_poor,  log_target_pdf, rwmh_prop_covar, num_gen= 500);
results_demcmc = run_demcmc( pop_init_poor,  log_target_pdf, num_gen= 500);
results_affine = run_affine_pop_mcmc(pop_init_poor,  log_target_pdf, num_gen= 500);

#Plot and inspect the evolution of the model parameters for a few chains from each simulation.
plot_trace(results_rwmh,1)
plot_trace(results_demcmc,1)
plot_trace(results_affine,1)

#Plot the resulting samples from each simulation, using the plot_sample function.  Overplot, samples taken just from the end of the chain.  For example,
plot_sample(results_rwmh,style="r.")
plot_sample(results_rwmh,style="b.", overplot=true, gen_start=101, gen_stop = 300)
plot_sample(results_rwmh,style="g.", overplot=true, gen_start=301)

#= 3.  Sampling from a Multi-modal Target Density
For real problems, there can be multiple significant modes in the posterior
 probability density.  Even if there aren’t, a scientists may not know that there
 isn’t a second mode.  In this exercise, we'll try sampling from a multi-modal
 target density.  Replace the log_target_pdf function with a version designed for
 a multi-modal target density.  For example,=#
tmp = reshape([0.1, 0.0, 0.0, 1.0], (2,2))
target2_covar = tmp'*tmp
target2_covar_fact = cholfact(target2_covar)

function log_target_pdf(theta::Array{Float64,1}, beta::Float64 = 1.0 )
  @assert(size(theta,1) == ndim)
  # This example calculates the density of a mixture of two Gaussian distributions
  # with means loc1 and loc2 and a common covariance matrix.
  # The fraction of the density from each mixture component is given by frac1 and frac2=1-frac1
  loc1 = [1.0, 0.0 ]
  loc2 = [-3.0, 3.0 ]
  frac1 = 1.0/2.0
  logfrac1 = log(frac1)
  logfrac2 = log(1.0 - frac1)
  delta = theta.-loc1
  # Calculate pdf of each component distribution
  logpdf1 = -0.5*( dot(delta,target_covar_fact\delta) +  2*logdet(target_covar_fact) + ndim*log(2pi) )
  delta = theta.-loc2
  logpdf2 = -0.5*( dot(delta,target2_covar_fact\delta) +  2*logdet(target2_covar_fact) + ndim*log(2pi) )
  # Weight each by the fraction it contributes to the mixture
  logpdf1 += logfrac1
  logpdf2 += logfrac2
  logpdf = log_sum_of_log_quantities(logpdf1,logpdf2)
  logpdf *= beta
  return logpdf
end

 #=Our population MCMC algorithms need an initial population of parameter values.
 This time, initialize the population drawn from a multivariate normal which isn’t
 centered on either of the modes of the target distribution.  For example, =#
 popsize = 16
 offset = 0.0
 scale = 1.0
 pop_init = Array(Float64,ndim,popsize);
 for i in 1:popsize
     pop_init[:,i] = offset .+ scale.* randn(ndim)
 end

#= First, try a IRWMH MCMC simulation.  Since we know our initial population
isn’t likely very good, we'll do two simulations.  First, we do one simulation
for the chains to "burn-in".=#
results_rwmh = run_indep_rwmh_mcmc(pop_init,  log_target_pdf, rwmh_prop_covar_try, num_gen= 500);

#=Then, run the simulations some more, picking up where you left off, but
otherwise discarding the samples from during the burn-in phase.  We’re hoping
that this second set of chains will be usable for inference.=#
results_rwmh = run_indep_rwmh_mcmc(results_rwmh["theta_last"],  log_target_pdf, rwmh_prop_covar_try, num_gen= 500);

#Similarly, run DE-MCMC simulations.
results_demcmc = run_demcmc( pop_init,  log_target_pdf, num_gen= 500);
results_demcmc = run_demcmc( results_demcmc["theta_last"],  log_target_pdf, num_gen= 500);

#And finally, let's run AIP-MCMC simulations
results_affine = run_affine_pop_mcmc( pop_init,  log_target_pdf, num_gen= 500);
results_affine = run_affine_pop_mcmc( results_affine["theta_last"],  log_target_pdf, num_gen= 500);

#Plot the results posterior samples from each simulation, using the plot_sample function.
plot_sample(results_rwmh,style="g.")
plot_sample(results_demcmc,style="r.", overplot=true)
plot_sample(results_affine,style="b.", overplot=true)

#=In this case, we know that the target density has two modes.  Inspecting the
above plots, identify the slope (a) and intercept (b) for a straight line that
would divide the two posterior samples.  Set the variable a and b to these values
and check that the line accurately separates points from the two modes.  =#

a = 2.5
b = 1.0
linex = linspace(-4.0,4.0,11)
liney = a.*linex.+b
plot(linex,liney,"b-")

#For each algorithm, inspect the trace plots to look for any indications of non-convergence.
plot_trace(results_rwmh,1)
plot_trace(results_rwmh,2)


plot_trace(results_demcmc,1)
plot_trace(results_demcmc,2)


plot_trace(results_affine,1)
plot_trace(results_affine,2)
