#test GAMC performance as a function of number of steps into chain
#for case of KOI1270 model

using Klara
using GAMCSampler

covTTV=readdlm("../outputs/KOI1270Cov.txt",',')
pmeans=readdlm("../outputs/KOI1270Means.txt",',')
pmeans=vec(pmeans)

pstart=readdlm("../outputs/KOI1270Last.txt",',')
pstart=vec(pstart)

covTTVhalf= ctranspose(chol(covTTV))
B=covTTVhalf
include("../models/KOI1270model.jl")
include("../utils/MCMCdiagnostics.jl")

minstep=0.618
fGAMC(k; x=minstep)= GAMC(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+k, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=x,
  minorscale=0.01,
  c=0.01
)

ntests=20
k_array= zeros(Int,ntests)
dur_array=zeros(Int,ntests)
dur_array[1]=1000
adder=1000
for i in 2:ntests
    k_array[i]= k_array[i-1] + (i-1)*adder
    dur_array[i] = i*adder
end

#measure efficiency in interval from n to n+1 th k value
#ie first effciency is labeled k=0 from k=0 - k=1000
p = BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model = likelihood_model(p, false)

zstart=to_z(pstart)
p0= Dict(:p=>zstart)

MCtuner=GAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false)
)

outopts= Dict{Symbol, Any}(:monitor=>[:value], :diagnostics=>[:accept])

times=Vector(ntests)
aclengths=Array{Float64}(ntests, length(pinit))
ess_array=Array{Float64}(ntests, length(pinit))
measure2= Vector(ntests)

for j in 1:ntests
    mcsampler=fGAMC(k_array[j])
    mcrange=BasicMCRange(nsteps=dur_array[j])
    job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
    tic()
    run(job)
    times[j]=toc()
    outval=output(job).value
    aclengths[j,:]=aclength(outval, threshold=0.1, maxit=10000, jump=1, useabs=true)
    ess_array[j,:]=dur_array[j]./aclengths[j,:]
    measure2[j]=minimum(ess_array[j,:])/times[j]
end
