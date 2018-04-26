#longChain.jl
#runs a longer MCMC chain
using Distributions,ForwardDiff, Klara
using MAMALASampler

include("MCMCdiagnostics.jl")
include("sinusoidFunctions.jl")
#=
k1=5.0
phi1=4*pi/5

k2=3.5
phi2=5*pi/9

pinit=[k1*cos(phi1), k1*sin(phi1), 3*pi/7, k2*cos(phi2),k2*sin(phi2),pi/4]
pguess=[-3.0, 4.0, 3*pi/7+1e-6, -0.5, 3.0, pi/4-1e-6] #further away requires longer burnin
=#
#pinit=[-3.5,4.3,1.0,0.8, 3*pi/7]
#pguess=[-3.0,4.0,0.9, 1.2, 3*pi/7 + 1e-6]
#trueData=readdlm("sinharmonicData.txt", ',')
pscale=0.001

scale=ones(10)
scale[5]=1000.0
scale[10]=1000.0
pinit=[-3.5,4.3,1.0,0.8, 3*pi/7, 5.0, 2.0, -0.5, 0.4, pi/3]
pguess=[-3.0,4.0,0.9, 1.2, (3*pi/7 + 1e-6), 4.0, 3.0, -0.7, 0.5,(pi/3 - 1e-6)]

zinit=pinit.*scale
zguess=pinit.*scale
trueData=readdlm("scaledtwosinharmonicData.txt", ',')

function plogtarget{T<:Number}(z::Vector{T})
  param=z./scale
  xarr=trueData[:,1]
  #y=sinharmonicmodel(xarr,param)
  y=twosinharmonicmodel(xarr,param)

  chisq=0.0
  for j in 1:length(xarr)
    chisq+= (y[j]-trueData[j,2])^2/trueData[j,3]^2
  end

  return -chisq/2.0
end

gconfig=ForwardDiff.GradientConfig(zinit)
function pgradlogtarget{T<:Number}(z::Vector{T})
    param=z./scale
    gstore=ForwardDiff.gradient(plogtarget,z,gconfig)
    return gstore #don't need to divide by scale b/c gradient is in terms of z
end

hconfig=ForwardDiff.HessianConfig(pinit)
function ptensorlogtarget{T<:Number}(z::Vector{T})
  param=z./scale
  hstore=ForwardDiff.hessian(plogtarget,z,hconfig)
  return hstore
end

p = BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model = likelihood_model(p, false)
p0= Dict(:p=>zguess)
mcrange= BasicMCRange(nsteps=50000, burnin=10000)
outopts= Dict{Symbol, Any}(:monitor=>[:value],:diagnostics =>[:accept])
minstep=0.7686246100397739
#mcsampler=SMMALA(minstep, H -> softabs(H,1000.0))
mcsampler=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, tot, 10.),
  transform=H -> softabs(H, 1000.),
  driftstep=minstep,
  minorscale=0.001,
  c=0.01
)

MCtuner=MAMALAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=true)
)
#MCtuner=VanillaMCTuner(verbose=true)
#MCtuner=AcceptanceRateMCTuner(0.6,verbose=true)
#maybe this is only meant for tuning MALA
job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
run(job)
out=output(job)
outval=out.value
outacc=out.diagnosticvalues
#=
writedlm("../../../Documents/Exoplanet_ttv_data/values_scaled2sinusoidMAMALAminstep.txt", outval, ",")
writedlm("../../../Documents/Exoplanet_ttv_data/accept_scaled2sinusoidMAMALAminstep.txt", outacc, ",")

using PyPlot
using PyCall
@pyimport corner
fig1=corner.corner(outval', labels=["1","2","3","4","5","6","7","8","9","10"])
savefig("../plots/sinusoid_fitting/MAMALA_minstep_scaled2sinusoid_5mil_10kBurnin.png")
=#
