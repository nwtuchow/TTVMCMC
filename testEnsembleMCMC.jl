#test ensemble MCMC
using PyPlot
include("../ensembleMCMC/popmcmc.jl")

covTTV=readdlm("pilotCov3.txt",',')
pmeans=readdlm("pilotMeans3.txt",',')
pmeans=vec(pmeans)

pstart=readdlm("pilotLast3.txt",',')
pstart=vec(pstart)

covTTVhalf= ctranspose(chol(covTTV))
B=covTTVhalf #sigma^(1/2)

include("TTVmodel3.jl")
include("MCMCdiagnostics.jl")

ndim= 10
popsize= floor(Int64,3*ndim)
pop_init = Array(Float64,ndim,popsize) #in z space

zstart=to_z(pstart)
for i in 1:popsize
    pop_init[:,i]= zstart+0.1*(2*randn(ndim)-1.0)
end

results_demcmc = run_demcmc( pop_init,  plogtarget, num_gen= 500)
plot_trace(results_demcmc,1)

results_affine = run_affine_pop_mcmc( pop_init,  plogtarget, num_gen= 500)
plot_trace(results_affine,1)
