using Klara
using MAMALASampler

ndim=12

include("MCMCdiagnostics.jl")

#=pmeans=zeros(ndim)
B=eye(ndim)
include("newSSMmodel.jl")

fmat= -ptensorlogtarget(zinit) #estimate of fisher information matrix
covguess=inv(fmat)
covguess= 0.5*(covguess+covguess')
B=chol(covguess)=#

covSSM=readdlm("pilotCovSSM3.txt",',')
pmeans=readdlm("pilotMeansSSM3.txt",',')
pmeans=vec(pmeans)

B=chol(covSSM)
include("newSSMmodel3.jl")

p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)

p0= Dict(:p=>zguess)

numsteps=15000000
burnin=1000000
mcrange= BasicMCRange(nsteps=numsteps,burnin=burnin, thinning=10)

outopts = Dict{Symbol, Any}(:monitor=>[:value,:logtarget],
  :diagnostics=>[:accept])

MCtuner=VanillaMCTuner(verbose=true)

minstep=1.93e-7
mcsampler=MALA(minstep)

job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
println("Starting:")
run(job)
out=output(job)
outval=out.value
outlogtarget=out.logtarget
outacc=out.diagnosticvalues


for j in 1:(size(outval)[2])
  outval[:,j]=B*outval[:,j]+pmeans
end

writedlm("../../../Documents/Exoplanet_ttv_data/values_newSSM_MALAminstep3.txt", outval, ",")
writedlm("../../../Documents/Exoplanet_ttv_data/accept_newSSM_MALAminstep3.txt", outacc, ",")
writedlm("../../../Documents/Exoplanet_ttv_data/logtarget_newSSM_MALAminstep3.txt", outlogtarget, ",")

using PyPlot
using PyCall
using LaTeXStrings
@pyimport corner
corner.corner(outval', labels=[L"\mathbf{p_1}",L"\mathbf{p_2}",L"\mathbf{p_3}",L"\mathbf{p_4}",L"\mathbf{p_5}",L"\mathbf{p_6}",L"\mathbf{p_7}",L"\mathbf{p_8}",L"\mathbf{p_9}",L"\mathbf{p_{10}}",L"\mathbf{p_{11}}",L"\mathbf{p_{12}}"],
quantiles=[0.16, 0.5, 0.84],
show_titles=true)
