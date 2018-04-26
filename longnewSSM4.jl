using Klara
using MAMALASampler

ndim=8

include("MCMCdiagnostics.jl")

covSSM=readdlm("pilotCovSSM4.txt",',')
pmeans=readdlm("pilotMeansSSM4.txt",',')
pmeans=vec(pmeans)

B=chol(covSSM)
include("newSSMmodel4.jl")

p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)

p0= Dict(:p=>zguess)

numsteps=5000000
burnin=500000
mcrange= BasicMCRange(nsteps=numsteps,burnin=burnin, thinning=10)

outopts = Dict{Symbol, Any}(:monitor=>[:value,:logtarget],
  :diagnostics=>[:accept])

#MCtuner=VanillaMCTuner(verbose=true)
MCtuner=MAMALAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=true)
)

minstep=1.32
mcsampler=MAMALA(
    update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, 50000, 10.),
    transform=H -> softabs(H, 1500.),
    driftstep=minstep,
    minorscale=0.001,
    c=0.01)

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

writedlm("../../../Documents/Exoplanet_ttv_data/values_newSSM_MAMALAminstep4.txt", outval, ",")
writedlm("../../../Documents/Exoplanet_ttv_data/accept_newSSM_MAMALAminstep4.txt", outacc, ",")
writedlm("../../../Documents/Exoplanet_ttv_data/logtarget_newSSM_MAMALAminstep4.txt", outlogtarget, ",")

using PyPlot
using PyCall
using LaTeXStrings
@pyimport corner
corner.corner(outval', labels=[L"\mathbf{p_1}",L"\mathbf{p_2}",L"\mathbf{p_3}",L"\mathbf{p_4}",L"\mathbf{p_5}",L"\mathbf{p_6}",L"\mathbf{p_7}",L"\mathbf{p_8}"],
quantiles=[0.16, 0.5, 0.84],
show_titles=true)
