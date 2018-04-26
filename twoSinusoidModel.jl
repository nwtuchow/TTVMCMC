# modelling two sinusoids and harmonics
using ForwardDiff

include("sinusoidFunctions.jl")

scale=ones(10)
scale[5]=1000.0
scale[10]=1000.0
pinit=[-3.5,4.3,1.0,0.8, 3*pi/7, 5.0, 2.0, -0.5, 0.4, pi/3]
pguess=[-3.0,4.0,0.9, 1.2, (3*pi/7 + 1e-6), 4.0, 3.0, -0.7, 0.5,(pi/3 - 1e-6)]

zinit=pinit.*scale
zguess=pguess.*scale
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
#=
numsteps=15
#driftsteps= logspace(-9.0,-7.0,numsteps) #for pscale=1 MALA
driftsteps= logspace(-5.0,0.7,numsteps) #for pscale=0.001

#no proposals accepted for steps >~0.0003
aclengths=Array{Float64}(numsteps, length(pinit))
ess_array=Array{Float64}(numsteps, length(pinit))
accrate=Vector{Float64}(numsteps)
p = BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model = likelihood_model(p, false)
p0= Dict(:p=>pguess)
mcrange= BasicMCRange(nsteps=50000, burnin=2500)
outopts= Dict{Symbol, Any}(:monitor=>[:value], :diagnostics=>[:accept])
#MCtuner=AcceptanceRateMCTuner(0.6,verbose=false)
#MCtuner=VanillaMCTuner(verbose=true)
MCtuner=MAMALAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=true)
)

for i in 1:numsteps
  #mcsampler=SMMALA(driftsteps[i],H -> softabs(H,1000.0))
  mcsampler=MAMALA(
    update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, tot, 10.),
    transform=H -> softabs(H, 1000.),
    driftstep=driftsteps[i],
    minorscale=0.001,
    c=0.01
  )
  job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
  println("Job: ", i," , Step size: ", driftsteps[i])
  run(job)
  outval=output(job).value
  acc=output(job).diagnosticvalues
  accrate[i]=mean(acc)
  aclengths[i,:]=aclength(outval, threshold=0.1, maxit=25000, jump=5)
  ess_array[i,:]=ess(output(job))
end

maxac=Vector(numsteps)
for i in 1:numsteps
    if(any(isnan(aclengths[i,:])) || accrate[i]<0.01) #only use ac lengths for runs that accept proposals
        maxac[i]=Inf
    else
        maxac[i]=maximum(aclengths[i,:])
    end
end

minind=indmin(maxac)
minstep=driftsteps[minind]

writedlm("../outputs/aclength_MAMALA_2sinusoid.txt", aclengths, ",")
writedlm("../outputs/ess_MAMALA_2sinusoid.txt", ess_array, ",")
writedlm("../outputs/accept_MAMALA_2sinusoid.txt", accrate, ",")
writedlm("../outputs/step_MAMALA_2sinusoid.txt", driftsteps, ",")

using Plots
plotly()

acplot=scatter(driftsteps,aclengths,
  layout=10,
  xaxis=( :log10),
  title=["p1" "p2" "p3" "p4" "p5" "p6" "p7" "p8" "p9" "p10"],
  leg=false)

essplot=scatter(driftsteps,ess_array,
  layout=10,
  xaxis=( :log10),
  title=["p1" "p2" "p3" "p4" "p5" "p6" "p7" "p8" "p9" "p10"],
  leg=false)

acceptplot= scatter(driftsteps, accrate,
  xaxis=:log10,
  xlabel="HMC step size",
  ylabel="Net acceptance rate",
  leg=false)
=#
