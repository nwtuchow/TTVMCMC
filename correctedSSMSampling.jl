#Testing MCMC in Klara
using Klara
using MAMALASampler

covTTV=readdlm("pilotCovCorrSSM.txt",',')
pmeans=readdlm("pilotMeansCorrSSM.txt",',')
pmeans=vec(pmeans)

pstart=readdlm("pilotLastCorrSSM.txt",',')
pstart=vec(pstart)

covTTVhalf= ctranspose(chol(covTTV))
B=covTTVhalf #sigma^(1/2)

include("correctedSSMmodel.jl")
include("MCMCdiagnostics.jl")

include("tuneSampler.jl")
fhmc1(x)=HMC(x,1)
fhmc2(x)=HMC(x,2)
fhmc3(x)=HMC(x,3)
fhmc5(x)=HMC(x,5)
fhmc7(x)=HMC(x,7)
fmala(x)=MALA(x)
fsmmala(x)=SMMALA(x, H -> simple_posdef(H, a=1500.))
fmamala1(x)=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=x,
  minorscale=0.01,
  c=0.01
)
fmamala2(x)=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+25000, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=x,
  minorscale=0.01,
  c=0.01
)
fmamala3(x)=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+50000, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=x,
  minorscale=0.01,
  c=0.01
)
fmamala4(x)=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+1000000, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=x,
  minorscale=0.01,
  c=0.01
)

nsamp=11
sampfuncs=[fhmc1,fhmc2,fhmc3,fhmc5,fhmc7,fmala,fsmmala,fmamala1,fmamala2,fmamala3,fmamala4]
useMAMALA=[false,false,false,false,false,false,false,true,true,true,true]
tune_arr=Vector(nsamp)

start=-1.0
stop=0.3

for q in 1:nsamp
    tune_arr[q]=tuneSampler(sampfuncs[q],plogtarget,pgradlogtarget,ptensorlogtarget,
        numtune=10,start=start,stop=stop,MAMALAtuner=useMAMALA[q])
end

###########################
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

times=Vector(nsamp)
aclengths=Array{Float64}(nsamp, length(pinit))
ess_array=Array{Float64}(nsamp, length(pinit))
accrate=Vector{Float64}(nsamp)

for i in 1:nsamp
    mcsampler=sampfuncs[i](tune_arr[i]["minstep"])
    if useMAMALA[i]
        MCtuner=MAMALAMCTuner(
          VanillaMCTuner(verbose=false),
          VanillaMCTuner(verbose=false),
          VanillaMCTuner(verbose=false)
        )
    else
        MCtuner=VanillaMCTuner(verbose=false)
    end
    job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
    tic()
    run(job)
    times[i]=toc()
    outval=output(job).value
    aclengths[i,:]=aclength(outval, threshold=0.1, maxit=10000, jump=1, useabs=true)
    #ess_array[i,:]=ess(output(job))
    ess_array[i,:]=nstep./aclengths[i,:]
    accrate[i]=acceptance(output(job))
end

measure1=Vector{Float64}(nsamp) #mean ess/time
measure2=Vector{Float64}(nsamp) #min ess/time
for j in 1:(nsamp)
  measure1[j]=mean(ess_array[j,:])/times[j]
  measure2[j]=minimum(ess_array[j,:]/times[j])
end
