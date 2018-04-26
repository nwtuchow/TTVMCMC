#compare efficiency for different samplers
using Klara
using MAMALASampler
include("MCMCdiagnostics.jl")

ndim=8
covSSM=readdlm("pilotCovSSM4.txt",',')
pmeans=readdlm("pilotMeansSSM4.txt",',')
pmeans=vec(pmeans)

B=chol(covSSM)
include("newSSMmodel4.jl")

p = BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model = likelihood_model(p, false)

p0= Dict(:p=>zguess)

mcrange= BasicMCRange(nsteps=1000000, burnin=100000)

outopts= Dict{Symbol, Any}(:monitor=>[:value], :diagnostics=>[:accept])

HMCminstep=0.00190
samplerHMC=HMC(HMCminstep,5)
MALAminstep=1.38e-5
samplerMALA=MALA(MALAminstep)
SMMALAminstep=1.32
samplerSMMALA=SMMALA(SMMALAminstep,H -> softabs(H,1500.0))
MAMALAminstep=0.848
samplerMAMALA=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, 50000, 10.),
  transform=H -> softabs(H, 1500.),
  driftstep=MAMALAminstep,
  minorscale=0.001,
  c=0.01
)

MAMALAminstep2=0.88
samplerMAMALA2=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+25000, 50000, 10.),
  transform=H -> softabs(H, 1500.),
  driftstep=MAMALAminstep,
  minorscale=0.001,
  c=0.01
)

MAMALAminstep3=0.88
samplerMAMALA3=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+50000, 50000, 10.),
  transform=H -> softabs(H, 1500.),
  driftstep=MAMALAminstep,
  minorscale=0.001,
  c=0.01
)

MAMALAminstep4=0.88
samplerMAMALA4=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+1000000, 50000, 10.),
  transform=H -> softabs(H, 1500.),
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

#=nsamp=7
samplerArray=[samplerHMC,samplerMALA,samplerSMMALA,samplerMAMALA, samplerMAMALA2,samplerMAMALA3,samplerMAMALA4]
tunerArray=[tuner1,tuner1,tuner1,tuner2,tuner2,tuner2,tuner2]
=#
nsamp=4
samplerArray=[samplerHMC,samplerMALA,samplerSMMALA,samplerMAMALA]
tunerArray=[tuner1,tuner1,tuner1,tuner2]

times=Vector(nsamp)
aclengths=Array{Float64}(nsamp, length(pinit))
ess_array=Array{Float64}(nsamp, length(pinit))
acrate=Array{Float64}(nsamp)

for i in 1:nsamp
  job=BasicMCJob(model,samplerArray[i],mcrange, p0, tuner=tunerArray[i], outopts=outopts)
  tic()
  run(job)
  times[i]=toc()
  outval=output(job).value
  #aclengths[i,:]=aclength(outval, threshold=0.1, maxit=7500, jump=5)
  ess_array[i,:]=ess(output(job))
  acrate[i]=acceptance(output(job))
end

measure1=Vector{Float64}(nsamp) #mean ess/time
measure2=Vector{Float64}(nsamp) #min ess/time
for j in 1:nsamp
  measure1[j]=mean(ess_array[j,:])/times[j]
  measure2[j]=minimum(ess_array[j,:]/times[j])
end
