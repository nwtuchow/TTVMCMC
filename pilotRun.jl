#pilot run
using Klara
using MAMALASampler
ndim=10
pmeans=zeros(ndim)
B=eye(ndim)
include("TTVmodel.jl")

hguess=softabs(ptensorlogtarget(zguess),1000.0)
covguess= (hguess) \ eye(ndim)
covguess= 0.5*(covguess+covguess')
B=chol(covguess)

include("TTVmodel.jl")

p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)

p0= Dict(:p=>zguess)

mcrange= BasicMCRange(nsteps=1000000,burnin=100000)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])

#MCtuner=VanillaMCTuner(verbose=true)
MCtuner=MAMALAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=true)
)

#mcsampler=MALA(1e-3) # for scaled
#mcsampler=MALA(1e-5)
mcsampler=MAMALA(
    update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+25000, 50000, 10.),
    transform=H -> softabs(H, 1500.),
    driftstep=0.1,
    minorscale=0.001,
    c=0.01)

job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)

println("Pilot Run:")
run(job)

outarr=output(job).value

#transform outarr to unscaled
for j in 1:(size(outarr)[2])
  outarr[:,j]=B*outarr[:,j]+pmeans
end

#outarr=inv(scale)*outarr

pilotmeans=zeros(10)
for i in 1:np
   pilotmeans[i]=mean(outarr[i,:])
end

pilotcov=samplecov(outarr) #unscaled parameter covariance

writedlm("pilotCov2.txt", pilotcov,",")
writedlm("pilotMeans2.txt", pilotmeans, ",")
