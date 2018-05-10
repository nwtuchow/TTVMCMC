#pilot run for new SSM reparameterized
using Klara
using GAMCSampler
ndim=12
pmeans=zeros(ndim)
B=eye(ndim)
include("correctedSSMmodel.jl")

fmat= -ptensorlogtarget(zguess) #estimate of fisher information matrix
covguess= fmat \ eye(ndim)
covguess= 0.5*(covguess+covguess')
B=ctranspose(chol(covguess))
pmeans[1]=ptrue[1]
pmeans[2]=ptrue[2]
pmeans[7]=ptrue[7]
pmeans[8]=ptrue[8]
include("correctedSSMmodel.jl")


p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)

p0= Dict(:p=>zguess)

mcrange= BasicMCRange(nsteps=500000)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])

#MCtuner=VanillaMCTuner(verbose=true)
#mcsampler=MALA(1e-2)
MCtuner=GAMCMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=true)
)


mcsampler=GAMC(
    update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, 50000, 10.),
    transform=H -> simple_posdef(H, a=1500.),
    driftstep=0.5,
    minorscale=0.01,
    c=0.01)

job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
println("Pilot Run:")
run(job)

outarr=output(job).value
outarrz=copy(outarr)
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
plast=outarr[:,end]

writedlm("pilotCovCorrSSM.txt", pilotcov,",")
writedlm("pilotMeansCorrSSM.txt", pilotmeans, ",")
writedlm("pilotLastCorrSSM.txt",plast, ",")
