#test ensemble MCMC efficiency

include("../utils/popmcmc.jl")

pmeans=zeros(ndim)
B=eye(ndim)

include("../models/SSMmodel.jl")
include("../utils/MCMCdiagnostics.jl")

ndim= 12
popsize= floor(Int64,3*ndim) #number of walkers

pop_init=readdlm("../outputs/ensembleLastSSM.txt",',')
numsteps=10000
tic()
results_demcmc = run_demcmc( pop_init,  plogtarget, num_gen= numsteps)
#results_demcmc= run_affine_pop_mcmc(pop_init,plogtarget, num_gen=numsteps)
times=toc()

outchains=results_demcmc["theta_all"]
accrate=sum(results_demcmc["accepts_generation"])/(sum(results_demcmc["accepts_generation"])+sum(results_demcmc["rejects_generation"]))

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

diagnostic_array=readdlm("../outputs/untransformedSSMdiagnostics.txt",',')
new_entry=["DEMCMC",NaN,times,measure1,measure2]
new_entry=reshape(new_entry, (1,5))
diagnostic_array=vcat(diagnostic_array,new_entry)

writedlm("../outputs/untransformedSSMdiagnostics.txt", diagnostic_array, ",")
