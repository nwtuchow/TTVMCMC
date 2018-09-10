#test ensemble MCMC efficiency
#for k307 inspired TTV model

include("../utils/popmcmc.jl")

covTTV=readdlm("../outputs/pilotCov3.txt",',')
pmeans=readdlm("../outputs/pilotMeans3.txt",',')
pmeans=vec(pmeans)


covTTVhalf= ctranspose(chol(covTTV))
B=covTTVhalf #sigma^(1/2)

include("../models/k307model.jl")
include("../utils/MCMCdiagnostics.jl")

ndim= 10
popsize= floor(Int64,3*ndim) #number of walkers

pop_init=readdlm("../outputs/ensembleLast.txt",',')
numsteps=10000
tic()
results_aimcmc = run_affine_pop_mcmc( pop_init,  plogtarget, num_gen= numsteps)
times=toc()

outchains=results_aimcmc["theta_all"]
accrate=sum(results_aimcmc["accepts_generation"])/(sum(results_aimcmc["accepts_generation"])+sum(results_aimcmc["rejects_generation"]))

aclengths=Array{Float64}(popsize,ndim)
ess_array=Array{Float64}(popsize,ndim)

for pop in 1:popsize
    aclengths[pop,:]=aclength(outchains[:,pop,:], threshold=0.1, maxit=10000, jump=1, useabs=true)
    ess_array[pop,:]=numsteps./aclengths[pop,:]
end

tot_ess=Vector{Float64}(ndim)

for i in 1:ndim
    tot_ess[i]=sum(ess_array[:,i])
end

measure1=mean(tot_ess)/times
measure2=minimum(tot_ess)/times

diagnostic_array=readdlm("../outputs/kepler307diagnostics2.txt",',')
new_entry=["AIMCMC",NaN,times,measure1,measure2]
new_entry=reshape(new_entry, (1,5))
diagnostic_array=vcat(diagnostic_array,new_entry)

writedlm("../outputs/kepler307diagnostics2.txt", diagnostic_array, ",")
