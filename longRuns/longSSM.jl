#long run for corrected SSM
using Klara

covTTV=readdlm("../outputs/pilotCovCorrSSM.txt",',')
pmeans=readdlm("../outputs/pilotMeansCorrSSM.txt",',')
pmeans=vec(pmeans)

pstart=readdlm("../outputs/pilotLastCorrSSM.txt",',')
pstart=vec(pstart)

covTTVhalf= ctranspose(chol(covTTV))
B=covTTVhalf #sigma^(1/2)

include("../models/SSMmodel.jl")
include("../utils/MCMCdiagnostics.jl")

p = BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model = likelihood_model(p, false)

zstart=to_z(pstart)
p0= Dict(:p=>zstart)

numsteps=2000000
#burnin=500000
mcrange= BasicMCRange(nsteps=numsteps, thinning=10)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])

MCtuner=VanillaMCTuner(verbose=true)

minstep=0.918
mcsampler=HMC(minstep,2)

job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
run(job)
out=output(job)
outval=out.value
outacc=out.diagnosticvalues

outvalz=copy(outval)

for j in 1:(size(outval)[2])
  outval[:,j]=B*outval[:,j]+pmeans
end

writedlm("../outputs/values_correctedSSM_HMC.txt", outval, ",")
writedlm("../outputs/accept_correctedSSM_HMC.txt", outacc, ",")

using PyPlot
using PyCall
using LaTeXStrings
@pyimport corner
corner.corner(outval', labels=[1,2,3,4,5,6,7,8,9,10,11,12],
quantiles=[0.16, 0.5, 0.84],
show_titles=true)

savefig("../outputs/HMC_correctedSSM_2mil.png")
