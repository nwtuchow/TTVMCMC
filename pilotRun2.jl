#pilot run
using Klara
using MAMALASampler
ndim=10
pmeans=zeros(ndim)
B=eye(ndim)
include("TTVmodel2.jl")

#=hguess=softabs(ptensorlogtarget(zguess),1000.0)
covguess= (hguess) \ eye(ndim)
covguess= 0.5*(covguess+covguess')
B=chol(covguess)

include("TTVmodel2.jl")=#

p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)

zerror=[2.4176e-5,  10.5, 783.999,  -0.0186066,   0.00391197,   1.087e-5,  13.0, 784.999,  -0.00952849,   0.0100977]
#p0= Dict(:p=>zguess)
p0=Dict(:p=>zerror)

mcrange= BasicMCRange(nsteps=50000,burnin=5000)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])

#MCtuner=VanillaMCTuner(verbose=true)
MCtuner=MAMALAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=true)
)

#mcsampler=HMC(1e-4,5)
#mcsampler=MALA(1e-3) # for scaled
#mcsampler=MALA(1e-9)
mcsampler=MAMALA(
    update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, 50000, 10.),
    transform=H -> simple_posdef(H, a=1000.),
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
