#Testing MCMC in Klara
using Klara
using MAMALASampler

covTTV=readdlm("NoisyCov.txt",',')
pmeans=readdlm("NoisyMeans.txt",',')
pmeans=vec(pmeans)

pstart=readdlm("NoisyLast.txt",',')
pstart=vec(pstart)

covTTVhalf= ctranspose(chol(covTTV))
B=covTTVhalf #sigma^(1/2)

include("NoisyTTVmodelOld.jl")
include("MCMCdiagnostics.jl")

p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)


zstart=to_z(pstart)
p0= Dict(:p=>zstart)

numsteps=5000000
#burnin=500000
mcrange= BasicMCRange(nsteps=numsteps, thinning=10)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])

MCtuner=VanillaMCTuner(verbose=true)

minstep=0.755
mcsampler=HMC(minstep,3)

job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
run(job)
out=output(job)
outval=out.value
outacc=out.diagnosticvalues

outvalz=copy(outval)

for j in 1:(size(outval)[2])
  outval[:,j]=B*outval[:,j]+pmeans
end

writedlm("../../../Documents/Exoplanet_ttv_data/values_NoisyKep307MAMALA.txt", outval, ",")
writedlm("../../../Documents/Exoplanet_ttv_data/accept_NoisyKep307MAMALA.txt", outacc, ",")

using PyPlot
using PyCall
using LaTeXStrings
@pyimport corner
corner.corner(outval', labels=[L"\mathbf{\mu_b}",L"\mathbf{P_b}",L"\mathbf{t_{i,b}}",L"\mathbf{k_b}",L"\mathbf{h_b}",L"\mathbf{\mu_c}",L"\mathbf{P_c}",L"\mathbf{t_{i,c}}",L"\mathbf{k_c}",L"\mathbf{h_c}"],
quantiles=[0.16, 0.5, 0.84],
show_titles=true)

savefig("../plots/cornerplots/HMC_NoisyTTV_5mil")
