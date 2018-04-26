#individual sampler efficiency
using Klara
#using MAMALASampler

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

#=include("tuneSampler.jl")
fhmc1(x)=HMC(x,1)
tune1=tuneSampler(fhmc1,plogtarget,pgradlogtarget,ptensorlogtarget,numtune=10,start=-1.0,stop=0.3)

minstep=tune1["minstep"]=#
minstep=0.822
samplerHMC2=HMC(minstep,2)

tuner1=VanillaMCTuner(verbose=false)
#=tuner2=MAMALAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false)
)=#

#aclengths=Vector(length(pinit))
#ess_array=Vector(length(pinit))

job=BasicMCJob(model,samplerHMC2,mcrange, p0, tuner=tuner1, outopts=outopts)
tic()
run(job)
times=toc()
outval=output(job).value
aclengths=aclength(outval, threshold=0.1, maxit=10000, jump=1, useabs=true)
#ess_array[i,:]=ess(output(job))
ess_array=nstep./aclengths
accrate=acceptance(output(job))

measure1=mean(ess_array)/times
measure2=minimum(ess_array/times)
