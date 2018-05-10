#kepler 57 pilot run
#julia 5 version
using Klara
using GAMCSampler
ndim=10
pmeans=zeros(ndim)
B=eye(ndim)
include("KOI1270Data_model.jl")

fmat= -ptensorlogtarget(zguess) #estimate of fisher information matrix
covguess= fmat \ eye(ndim)
covguess= 0.5*(covguess+covguess')
for k in 1:ndim
    B[k,k]=sqrt(abs(covguess[k,k]))
end
#B=ctranspose(chol(covguess))
pmeans=copy(pinit)
include("KOI1270Data_modelOld.jl")

p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)
p0= Dict(:p=>zguess)

mcrange= BasicMCRange(nsteps=500000,burnin=50000)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])

#MCtuner=VanillaMCTuner(verbose=true)
MCtuner=GAMCMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=true)
)

#mcsampler=SMMALA(0.3, H -> simple_posdef(H, a=1500.))
mcsampler=GAMC(
    update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, 50000, 10.),
    transform=H -> simple_posdef(H, a=1500.),
    driftstep=0.3,
    minorscale=0.01,
    c=0.01)


job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)

println("Pilot Run:")
run(job)

outarr=output(job).value

for j in 1:(size(outarr)[2])
  outarr[:,j]=to_p(outarr[:,j])
end

pilotmeans=zeros(np)
for i in 1:np
   pilotmeans[i]=mean(outarr[i,:])
end

pilotcov=samplecov(outarr) #unscaled parameter covariance
plast=outarr[:,end]

writedlm("KOI1270Cov2.txt", pilotcov,",")
writedlm("KOI1270Means2.txt", pilotmeans, ",")
writedlm("KOI1270Last2.txt", plast, ",")
