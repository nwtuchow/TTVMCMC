#julia 6 version
#tune samplers and compare efficiency for KOI248 model
using Klara
using GAMCSampler

covTTV=readdlm("KOI248Cov.txt",',')
pmeans=readdlm("KOI248Means.txt",',')
pmeans=vec(pmeans)

pstart=readdlm("KOI248Last.txt",',')
pstart=vec(pstart)

covTTVhalf= ctranspose(chol(covTTV))
B=covTTVhalf #sigma^(1/2)

include("KOI248model.jl")
include("MCMCdiagnostics.jl")

include("tuneSampler.jl")
fhmc1(x)=HMC(x,1)
fhmc2(x)=HMC(x,2)
fhmc3(x)=HMC(x,3)
fhmc5(x)=HMC(x,5)
fhmc7(x)=HMC(x,7)
fmala(x)=MALA(x)
fsmmala(x)=SMMALA(x, H -> simple_posdef(H, a=1500.))
fgamc1(x)=GAMC(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=x,
  minorscale=0.01,
  c=0.01
)
fgamc2(x)=GAMC(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+25000, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=x,
  minorscale=0.01,
  c=0.01
)
fgamc3(x)=GAMC(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+50000, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=x,
  minorscale=0.01,
  c=0.01
)
fgamc4(x)=GAMC(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+1000000, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=x,
  minorscale=0.01,
  c=0.01
)

nsamp=11
sampnames=["HMC1", "HMC2","HMC3","HMC5","HMC7","MALA","SMMALA","GAMC(i=0)","GAMC(i=25000)","GAMC(i=50000)","GAMC(i=1e6)"]
sampfuncs=[fhmc1,fhmc2,fhmc3,fhmc5,fhmc7,fmala,fsmmala,fgamc1,fgamc2,fgamc3,fgamc4]
useGAMC=[false,false,false,false,false,false,false,true,true,true,true]
tune_arr=Vector(nsamp)
minsteps=Vector(nsamp)
#=
start=-3.0
stop=0.7

for q in 1:nsamp
    tune_arr[q]=tuneSampler(sampfuncs[q],plogtarget,pgradlogtarget,ptensorlogtarget,
        numtune=10,start=start,stop=stop,GAMCtuner=useGAMC[q],narrow=2)
    minsteps[q]=tune_arr[q]["minstep"]
end
=#
minsteps=[0.0544,0.0584,0.0421,0.0490,0.0397,0.00538,0.283,0.868,0.140,0.349,0.349]
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
    mcsampler=sampfuncs[i](minsteps[i])
    if useGAMC[i]
        MCtuner=GAMCMCTuner(
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

diagnostic_array=hcat(sampnames,minsteps,times,measure1,measure2)
writedlm("KOI248diagnostics.txt", diagnostic_array, ",")
