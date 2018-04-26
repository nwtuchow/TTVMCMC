#compare efficiency for different samplers
using Klara
using MAMALASampler

covTTV=readdlm("pilotCov3.txt",',')
pmeans=readdlm("pilotMeans3.txt",',')
pmeans=vec(pmeans)

pstart=readdlm("pilotLast3.txt",',')
pstart=vec(pstart)

covTTVhalf= ctranspose(chol(covTTV))
B=covTTVhalf #sigma^(1/2)

include("TTVmodel3.jl")
include("MCMCdiagnostics.jl")

p = BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model = likelihood_model(p, false)

zstart=to_z(pstart)
p0= Dict(:p=>zstart)

nstep=10000
mcrange= BasicMCRange(nsteps=nstep)

outopts= Dict{Symbol, Any}(:monitor=>[:value], :diagnostics=>[:accept])

HMCminstep1=0.918
samplerHMC1=HMC(HMCminstep1,1)

HMCminstep2=0.822
samplerHMC2=HMC(HMCminstep2,2)

HMCminstep3=0.717
samplerHMC3=HMC(HMCminstep3,3)

HMCminstep5=0.736
samplerHMC5=HMC(HMCminstep5,5)

HMCminstep7=1.03
samplerHMC7=HMC(HMCminstep7,7)

MALAminstep=0.880
samplerMALA=MALA(MALAminstep)
SMMALAminstep=0.464
samplerSMMALA=SMMALA(SMMALAminstep,H -> simple_posdef(H, a=1500.))
MAMALAminstep=0.880
samplerMAMALA=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=MAMALAminstep,
  minorscale=0.01,
  c=0.01
)

samplerMAMALA2=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+25000, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=MAMALAminstep,
  minorscale=0.01,
  c=0.01
)

samplerMAMALA3=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+50000, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=MAMALAminstep,
  minorscale=0.01,
  c=0.01
)

samplerMAMALA4= MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+1000000, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=MAMALAminstep,
  minorscale=0.01,
  c=0.01
)

tuner1=VanillaMCTuner(verbose=false)
tuner2=MAMALAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false)
)

nsamp=11
samplerArray=[samplerHMC1,samplerHMC2, samplerHMC3,samplerHMC5,samplerHMC7, samplerMALA,samplerSMMALA,samplerMAMALA, samplerMAMALA2, samplerMAMALA3,samplerMAMALA4]
tunerArray=[tuner1,tuner1,tuner1,tuner1, tuner1, tuner1,tuner1,tuner2,tuner2,tuner2,tuner2]

times=Vector(nsamp+1)
aclengths=Array{Float64}(nsamp+1, length(pinit))
ess_array=Array{Float64}(nsamp+1, length(pinit))
accrate=Vector{Float64}(nsamp+1)

for i in 1:nsamp
  job=BasicMCJob(model,samplerArray[i],mcrange, p0, tuner=tunerArray[i], outopts=outopts)
  tic()
  run(job)
  times[i]=toc()
  outval=output(job).value
  aclengths[i,:]=aclength(outval, threshold=0.1, maxit=10000, jump=1, useabs=true)
  #ess_array[i,:]=ess(output(job))
  ess_array[i,:]=nstep./aclengths[i,:]
  accrate[i]=acceptance(output(job))
end

include("../ensembleMCMC/popmcmc.jl")

ndim= 10
popsize= floor(Int64,3*ndim) #number of walkers
pop_init=readdlm("ensembleLast.txt",',')

tic()
results_demcmc = run_demcmc( pop_init,  plogtarget, num_gen= nstep)
times[nsamp+1]=toc()

outchains=results_demcmc["theta_all"]
accrate[nsamp+1]=sum(results_demcmc["accepts_generation"])/(sum(results_demcmc["accepts_generation"])+sum(results_demcmc["rejects_generation"]))





pop_aclengths=Array{Float64}(popsize,ndim)
pop_ess_array=Array{Float64}(popsize,ndim)

for pop in 1:popsize
    pop_aclengths[pop,:]=aclength(outchains[:,pop,:], threshold=0.1, maxit=10000, jump=1, useabs=true)
    pop_ess_array[pop,:]=nstep./aclengths[pop,:]
end

for i in 1:ndim
    ess_array[nsamp+1,i]=sum(pop_ess_array[:,i])
    aclengths[nsamp+1,i]=nstep/ess_array[nsamp+1,i]
end


measure1=Vector{Float64}(nsamp+1) #mean ess/time
measure2=Vector{Float64}(nsamp+1) #min ess/time
for j in 1:(nsamp+1)
  measure1[j]=mean(ess_array[j,:])/times[j]
  measure2[j]=minimum(ess_array[j,:]/times[j])
end
