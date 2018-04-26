#pilot run for new SSM
#needs initial B and pmeans
include("newSSMmodel.jl")

p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)

p0= Dict(:p=>zguess)

mcrange= BasicMCRange(nsteps=100000,burnin=5000)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])

MCtuner=VanillaMCTuner(verbose=true)

mcsampler=MALA(1e-4)


job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)

println("Pilot Run:")
run(job)

outarr=output(job).value

#transform outarr to unscaled
for j in 1:(size(outarr)[2])
  outarr[:,j]=B*outarr[:,j]+pmeans
end

#outarr=inv(scale)*outarr

pilotmeans=zeros(ndim)
for i in 1:ndim
   pilotmeans[i]=mean(outarr[i,:])
end

pilotcov=samplecov(outarr) #unscaled parameter covariance

writedlm("pilotCovSSM.txt", pilotcov,",")
writedlm("pilotMeansSSM.txt", pilotmeans, ",")
