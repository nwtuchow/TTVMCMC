#long chain for KOI 1270 model using DEMCMC
include("../ensembleMCMC/popmcmc.jl")

covTTV=readdlm("KOI1270Cov.txt",',')
pmeans=readdlm("KOI1270Means.txt",',')
pmeans=vec(pmeans)

pstart=readdlm("KOI1270Last.txt",',')
pstart=vec(pstart)

covTTVhalf= ctranspose(chol(covTTV))
B=covTTVhalf #sigma^(1/2)

include("KOI1270modelOld.jl")
include("MCMCdiagnostics.jl")

ndim= 10
popsize= floor(Int64,3*ndim)
pop_init=readdlm("KOI1270ensembleLast.txt",',')

numsteps=150000
results_demcmc = run_demcmc( pop_init,  plogtarget, num_gen= numsteps)

pop_tot=results_demcmc["theta_all"]

outvalz=pop_tot[:,1,:]
for i in 2:popsize
    outvalz=cat(2,outvalz,pop_tot[:,i,:])
end

outval=copy(outvalz)
for j in 1:(size(outvalz)[2])
    outval[:,j]=to_p(outvalz[:,j])
end

outacc=results_demcmc["accepts_generation"]
writedlm("../../../Documents/Exoplanet_ttv_data/values_KOI1270MAMALA.txt", outval, ",")
writedlm("../../../Documents/Exoplanet_ttv_data/acceptgen_KOI1270MAMALA.txt", outacc, ",")

using PyPlot
using PyCall
using LaTeXStrings
@pyimport corner
corner.corner(outval', labels=[L"\mathbf{\mu_b}",L"\mathbf{P_b}",L"\mathbf{t_{i,b}}",L"\mathbf{k_b}",L"\mathbf{h_b}",L"\mathbf{\mu_c}",L"\mathbf{P_c}",L"\mathbf{t_{i,c}}",L"\mathbf{k_c}",L"\mathbf{h_c}"],
quantiles=[0.16, 0.5, 0.84],
show_titles=true)

savefig("../plots/cornerplots/DEMCMC_KOI1270_150000_30walker.png")
