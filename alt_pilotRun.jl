#pilot run
using Klara
using MAMALASampler
ndim=6
#pmeans=zeros(ndim)
#B=eye(ndim)
#=covTTV=readdlm("pilotCov2.txt",',')
pmeans=readdlm("pilotMeans2.txt",',')
pmeans=vec(pmeans)

covTTVhalf= chol(covTTV)
B=covTTVhalf #sigma^(1/2)
=#
include("TTVmodel3.jl")

p= BasicContMuvParameter(:p,
  logtarget=plogtarget_th,
  gradlogtarget=pgradlogtarget_th,
  tensorlogtarget=ptensorlogtarget_th)

model= likelihood_model(p, false)

p0= Dict(:p=>thguess)

mcrange= BasicMCRange(nsteps=300000,burnin=30000)

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
    driftstep=0.1,
    minorscale=0.01,
    c=0.01)

job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)

println("Pilot Run:")
run(job)

outarr=output(job).value

#transform outarr to unscaled
#=for j in 1:(size(outarr)[2])
  outarr[:,j]=B*outarr[:,j]+pmeans
end=#

#outarr=inv(scale)*outarr

pilotmeans=zeros(ndim)
for i in 1:ndim
   pilotmeans[i]=mean(outarr[i,:])
end

pilotcov=samplecov(outarr) #unscaled parameter covariance

writedlm("alt_pilotCov.txt", pilotcov,",")
writedlm("alt_pilotMeans.txt", pilotmeans, ",")
