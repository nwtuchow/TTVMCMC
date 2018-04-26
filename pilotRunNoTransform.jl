#pilot run
using Klara
using MAMALASampler
ndim=10
pmeans=zeros(ndim)
B=eye(ndim)
#covTTV=readdlm("pilotCov3.txt",',')
#pmeans=readdlm("pilotMeans3.txt",',')
#pmeans=vec(pmeans)

#covTTVhalf= chol(covTTV)
#B=covTTVhalf #sigma^(1/2)

include("TTVmodel3.jl")

#=hguess=softabs(ptensorlogtarget(zguess),1000.0)
covguess= (hguess) \ eye(ndim)
covguess= 0.5*(covguess+covguess')
B=ctranspose(chol(covguess))

include("TTVmodel2.jl")=#

p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)
p0= Dict(:p=>zguess)
#p0=Dict(:p=>zerror)

mcrange= BasicMCRange(nsteps=500000,burnin=50000)

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
    transform=H -> simple_posdef(H, a=1500.),
    driftstep=0.316,
    minorscale=0.01,
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
plast=outarr[:,end]

writedlm("pilotCovNoTransform.txt", pilotcov,",")
writedlm("pilotMeansNoTransform.txt", pilotmeans, ",")
writedlm("pilotLastNoTransform.txt", plast, ",")
