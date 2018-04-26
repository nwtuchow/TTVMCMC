#pilot run for new SSM reparameterized
using Klara
ndim=12
pmeans=zeros(ndim)
B=eye(ndim)
include("newSSMmodel3.jl")

fmat= -ptensorlogtarget(zinit) #estimate of fisher information matrix
covguess= fmat \ eye(ndim)
covguess= 0.5*(covguess+covguess')
B=chol(covguess)
include("newSSMmodel3.jl")

p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)

p0= Dict(:p=>zguess)

mcrange= BasicMCRange(nsteps=50000,burnin=5000)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])

MCtuner=VanillaMCTuner(verbose=true)

mcsampler=SMMALA(7e-1,H -> softabs(H,1500.0))


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

writedlm("pilotCovSSM3.txt", pilotcov,",")
writedlm("pilotMeansSSM3.txt", pilotmeans, ",")
