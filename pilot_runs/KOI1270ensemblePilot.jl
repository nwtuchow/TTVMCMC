#burnin ensembleMCMC for KOI 1270 model
include("../utils/popmcmc.jl")

covTTV=readdlm("../outputs/KOI1270Cov.txt",',')
pmeans=readdlm("../outputs/KOI1270Means.txt",',')
pmeans=vec(pmeans)

pstart=readdlm("../outputs/KOI1270Last.txt",',')
pstart=vec(pstart)

covTTVhalf= ctranspose(chol(covTTV))
B=covTTVhalf #sigma^(1/2)

include("../models/KOI1270model.jl")
include("../utils/MCMCdiagnostics.jl")

ndim= 10
popsize= floor(Int64,3*ndim)
pop_init = Array{Float64}(ndim,popsize)

testDist=MvNormal(pmeans,covTTV) #distribution for untransformed

for i in 1:popsize
    pop_init[:,i]=to_z(rand(testDist))
end

numsteps=50000
results_demcmc = run_demcmc( pop_init,  plogtarget, num_gen= numsteps)

pop_last=results_demcmc["theta_last"]
writedlm("../outputs/KOI1270ensembleLast.txt",pop_last,",")
