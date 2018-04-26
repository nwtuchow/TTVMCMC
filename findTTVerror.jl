#try to identify where chains crash
using Klara
using MAMALASampler


covTTV=readdlm("pilotCov3.txt",',')
pmeans=readdlm("pilotMeans3.txt",',')
pmeans=vec(pmeans)

covTTVhalf= ctranspose(chol(covTTV))
B=covTTVhalf #sigma^(1/2)


include("TTVmodel3.jl")
include("MCMCdiagnostics.jl")

lastchain=readdlm("../../../Documents/Exoplanet_ttv_data/values_transformedTTVFasterMAMALAminstep.txt",',')
plast=lastchain[:,end]
zlast=to_z(plast)
#=
lastcov=samplecov(lastchain)
B=chol(lastcov)

lastmeans=zeros(10)
for i in 1:10
   lastmeans[i]=mean(lastchain[i,:])
end
pmeans=lastmeans

include("TTVmodel3.jl")
lastchainz=copy(lastchain)
for n in 1:(size(lastchain)[2])
    lastchainz[:,n]=to_z(lastchain[:,n])
end
=#
p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)

p0= Dict(:p=>zlast)

mcrange=BasicMCRange(nsteps=50000)

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget,:tensorlogtarget],
  :diagnostics=>[:accept])

MCtuner=MAMALAMCTuner(
    VanillaMCTuner(verbose=false),
    VanillaMCTuner(verbose=false),
    VanillaMCTuner(verbose=true)
)

minstep=0.316
mcsampler=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+3212270, div(tot,100), 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=minstep,
  minorscale=0.01,
  c=0.01
)

job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
run(job)
out=output(job)
outval=out.value
outacc=out.diagnosticvalues

for j in 1:(size(outval)[2])
  outval[:,j]=B*outval[:,j]+pmeans
end
