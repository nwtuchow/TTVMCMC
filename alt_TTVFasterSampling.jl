#Testing MCMC in Klara
using Klara
using MAMALASampler

covTTV=readdlm("pilotCov3.txt",',')
pmeans=readdlm("pilotMeans3.txt",',')
pmeans=vec(pmeans)

covTTVhalf= chol(covTTV)
B=covTTVhalf #sigma^(1/2)

include("TTVmodel3.jl")
include("MCMCdiagnostics.jl")

p= BasicContMuvParameter(:p,
  logtarget=plogtarget_th,
  gradlogtarget=pgradlogtarget_th,
  tensorlogtarget=ptensorlogtarget_th)

model= likelihood_model(p, false)

p0= Dict(:p=>thguess)

mcrange= BasicMCRange(nsteps=50000,burnin=2500)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])

#MCtuner=AcceptanceRateMCTuner(0.6,verbose=true)
#MCtuner=VanillaMCTuner(verbose=true)
MCtuner=MAMALAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=true)
)

numsteps=10
driftsteps= logspace(-2.0,0.7,numsteps)
aclengths=Array{Float64}(numsteps, length(thinit))
ess_array=Array{Float64}(numsteps, length(thinit))
accrate=Vector{Float64}(numsteps)

for i in 1:numsteps
  #mcsampler=MALA(driftsteps[i])
  #mcsampler=HMC(driftsteps[i],5)
  #mcsampler=SMMALA(driftsteps[i], H -> softabs(H, 1000.))
  mcsampler=MAMALA(
    update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, tot, 10.),
    transform=H -> simple_posdef(H, a=1500.),
    driftstep=driftsteps[i],
    minorscale=0.01,
    c=0.01
  )
  job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
  println("Job: ", i," , Step size: ", driftsteps[i])
  run(job)
  outval=output(job).value
  acc=output(job).diagnosticvalues
  accrate[i]=mean(acc)
  aclengths[i,:]=aclength(outval, threshold=0.1, maxit=20000, jump=5)
  ess_array[i,:]=ess(output(job))
end
#=
writedlm("../outputs/aclength_SMMALA_transformedTTVFaster.txt", aclengths, ",")
writedlm("../outputs/ess_SMMALA_transformedTTVFaster.txt", ess_array, ",")
writedlm("../outputs/accept_SMMALA_transformedTTVFaster.txt", accrate, ",")
writedlm("../outputs/step_SMMALA_transformedTTVFaster.txt", driftsteps, ",")
=#
maxac=Vector(numsteps)
miness=Vector(numsteps)
for i in 1:numsteps
    if(any(isnan(aclengths[i,:])))
        maxac[i]=Inf
    else
        maxac[i]=maximum(aclengths[i,:])
    end
    miness[i]=minimum(ess_array[i,:])
end

minind=indmin(maxac)
minstep=driftsteps[minind]

maxind=indmax(miness)
maxstep=driftsteps[maxind]

using Plots
plotly()

acplot=scatter(driftsteps,aclengths,
  layout=6,
  xaxis=( :log10),
  title=["mu_b" "k_b" "h_b" "mu_c" "k_c" "h_c"],
  leg=false)

essplot=scatter(driftsteps,ess_array,
  layout=6,
  xaxis=( :log10),
  title=["mu_b" "k_b" "h_b" "mu_c" "k_c" "h_c"],
  leg=false)

acceptplot= scatter(driftsteps, accrate,
  xaxis=:log10,
  xlabel="Step size",
  ylabel="Net acceptance rate",
  leg=false)
