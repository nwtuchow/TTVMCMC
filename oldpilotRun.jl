#pilot run
using Klara
include("oldTTVmodel.jl")

p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)

p0= Dict(:p=>zguess)

mcrange= BasicMCRange(nsteps=15000,burnin=5000)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])

MCtuner=VanillaMCTuner(verbose=true)

mcsampler=MALA(1e-3)


job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)

println("Pilot Run:")
run(job)

outarr=output(job).value

#transform outarr to unscaled
outarr=inv(scale)*outarr

pmeans=zeros(Float64,np)
for i in 1:np
   pmeans[i]=mean(outarr[i,:])
end

covTTV=samplecov(outarr) #unscaled parameter covariance
