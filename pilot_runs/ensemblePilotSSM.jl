#burnin ensembleMCMC for corrected SSM

include("../utils/popmcmc.jl")

covTTV=readdlm("../outputs/pilotCovCorrSSM.txt",',')
pmeans=readdlm("../outputs/pilotMeansCorrSSM.txt",',')
pmeans=vec(pmeans)

pstart=readdlm("../outputs/pilotLastCorrSSM.txt",',')
pstart=vec(pstart)

covTTVhalf= ctranspose(chol(covTTV))
B=covTTVhalf #sigma^(1/2)

include("../models/SSMmodel.jl")
include("../utils/MCMCdiagnostics.jl")

ndim= 12
popsize= floor(Int64,3*ndim)
pop_init = Array{Float64}(ndim,popsize)

testDist=MvNormal(pmeans,covTTV) #distribution for untransformed

for i in 1:popsize
    pop_init[:,i]=to_z(rand(testDist))
end

numsteps=50000
results_demcmc = run_demcmc( pop_init,  plogtarget, num_gen= numsteps)

pop_last=results_demcmc["theta_last"]
writedlm("../outputs/ensembleLastSSM.txt",pop_last,",")
