#long chain for KOI 1270 model using GAMC
using Klara
using GAMCSampler

covTTV=readdlm("../outputs/KOI1270Cov.txt",',')
pmeans=readdlm("../outputs/KOI1270Means.txt",',')
pmeans=vec(pmeans)

pstart=readdlm("../outputs/KOI1270Last.txt",',')
pstart=vec(pstart)

covTTVhalf= ctranspose(chol(covTTV))
B=covTTVhalf #sigma^(1/2)

include("../models/KOI1270model.jl")
include("../utils/MCMCdiagnostics.jl")

p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)

zstart=to_z(pstart)
p0= Dict(:p=>zstart)

ndim= 10
numsteps=5000000

mcrange= BasicMCRange(nsteps=numsteps, thinning=10)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])

MCtuner=GAMCTuner(
    VanillaMCTuner(verbose=false),
    VanillaMCTuner(verbose=false),
    VanillaMCTuner(verbose=true)
)

minstep= 0.612

mcsampler=GAMC(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, 50000, 10.),
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

out_quantiles= cornerUncertainty(outval)

writedlm("../../Exoplanet_ttv_data/values_KOI1270GAMC.txt", outval, ",")
writedlm("../../Exoplanet_ttv_data/accept_KOI1270GAMC.txt", outacc, ",")
