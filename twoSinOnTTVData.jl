#using 2 sin Harmonic model on data from ttv faster optimization

using Klara
using MAMALASampler
ndim=9
pmeans=zeros(ndim)
scale=eye(ndim)

include("twoSinModelTTVFasterData.jl")
include("MCMCdiagnostics.jl")

p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)

p0= Dict(:p=>zguess)

numsteps=50000
burnin=2500
mcrange= BasicMCRange(nsteps=numsteps,burnin=burnin, thinning=1)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])


numsteps=10
driftsteps= logspace(-4.0,0.5,numsteps)
aclengths=Array{Float64}(numsteps, length(pinit))
ess_array=Array{Float64}(numsteps, length(pinit))
accrate=Vector{Float64}(numsteps)

MCtuner=VanillaMCTuner(verbose=true)
#MCtuner=MAMALAMCTuner(
#  VanillaMCTuner(verbose=false),
#  VanillaMCTuner(verbose=false),
#  VanillaMCTuner(verbose=true)
#)

#=
minstep=1.0
mcsampler=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, div(tot,100), 10.),
  transform=H -> softabs(H, 1000.),
  driftstep=minstep,
  minorscale=0.001,
  c=0.01
)

job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
run(job)
out=output(job)
outval=out.value
outacc=out.diagnosticvalues

outval=invscale*outval

using PyPlot
using PyCall
using LaTeXStrings
@pyimport corner
corner.corner(outval',labels=[L"\mathbf{p_1}", L"\mathbf{p_2}", L"\mathbf{p_3}", L"\mathbf{p_4}", L"\mathbf{p_5}", L"\mathbf{p_6}", L"\mathbf{p_7}", L"\mathbf{p_8}", L"\mathbf{p_9}"],
quantiles=[0.16, 0.5, 0.84],
show_titles=true,
title_fmt=".4f")


writedlm("../../../Documents/Exoplanet_ttv_data/values_twoSinOnTTVFasterMAMALAminstep.txt", outval, ",")
writedlm("../../../Documents/Exoplanet_ttv_data/accept_twoSinOnTTVFasterMAMALAminstep.txt", outacc, ",")
=#

for i in 1:numsteps
#mcsampler=MALA(1e-8)
#=mcsampler=MAMALA(,
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, tot, 10.),
  transform=H -> softabs(H, 1000.),
  driftstep=driftsteps[i],
  minorscale=0.001,
  c=0.01
)=#
mcsampler=SMMALA(driftsteps[i], H -> softabs(H,1000.0))

job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
println("Job: ", i," , Step size: ", driftsteps[i])
run(job)
out=output(job)
outval=out.value
outacc=out.diagnosticvalues
accrate[i]=mean(outacc)
aclengths[i,:]=aclength(outval, threshold=0.1, maxit=20000, jump=5)
ess_array[i,:]=ess(output(job))
end


#outval=invscale*outval

#testcov=samplecov(outval)

maxac=Vector(numsteps)
for i in 1:numsteps
    if(any(isnan(aclengths[i,:])))
        maxac[i]=Inf
    else
        maxac[i]=maximum(aclengths[i,:])
    end
end

minind=indmin(maxac)
minstep=driftsteps[minind]

using Plots
plotly()

acplot=Plots.scatter(driftsteps,aclengths,
  layout=9,
  xaxis=( :log10),
  title=["1" "2" "3" "4" "5" "6" "7" "8" "9"],
  leg=false)

essplot=Plots.scatter(driftsteps,ess_array,
  layout=9,
  xaxis=( :log10),
  title=["1" "2" "3" "4" "5" "6" "7" "8" "9"],
  leg=false)

acceptplot= Plots.scatter(driftsteps, accrate,
  xaxis=:log10,
  xlabel="Step size",
  ylabel="Net acceptance rate",
  leg=false)
