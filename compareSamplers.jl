#compare efficiency for different samplers
using Klara
using MAMALASampler

include("MCMCdiagnostics.jl")
include("twoSinusoidModel.jl")

p = BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model = likelihood_model(p, false)

p0= Dict(:p=>zguess)

mcrange= BasicMCRange(nsteps=10000, burnin=2500)

outopts= Dict{Symbol, Any}(:monitor=>[:value], :diagnostics=>[:accept])

HMCminstep=0.021544346900318832
samplerHMC=HMC(HMCminstep,5)
MALAminstep=0.00035938136638046257
samplerMALA=MALA(MALAminstep)
SMMALAminstep=0.848342898244072
samplerSMMALA=SMMALA(SMMALAminstep,H -> softabs(H,1000.0))
MAMALAminstep=0.7686246100397739
samplerMAMALA=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, tot, 10.),
  transform=H -> softabs(H, 1000.),
  driftstep=MAMALAminstep,
  minorscale=0.001,
  c=0.01
)

tuner1=VanillaMCTuner(verbose=false)
tuner2=MCtuner=MAMALAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false)
)

nsamp=4
samplerArray=[samplerHMC,samplerMALA,samplerSMMALA,samplerMAMALA]
tunerArray=[tuner1,tuner1,tuner1,tuner2]

times=Vector(nsamp)
aclengths=Array{Float64}(nsamp, length(pinit))
ess_array=Array{Float64}(nsamp, length(pinit))

for i in 1:nsamp
  job=BasicMCJob(model,samplerArray[i],mcrange, p0, tuner=tunerArray[i], outopts=outopts)
  tic()
  run(job)
  times[i]=toc()
  outval=output(job).value
  aclengths[i,:]=aclength(outval, threshold=0.1, maxit=7500, jump=5)
  ess_array[i,:]=ess(output(job))
end

measure1=Vector{Float64}(nsamp) #mean ess/time
measure2=Vector{Float64}(nsamp) #min ess/time
for j in 1:nsamp
  measure1[j]=mean(ess_array[j,:])/times[j]
  measure2[j]=minimum(ess_array[j,:]/times[j])
end
