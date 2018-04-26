#Testing MCMC in Klara
using Klara
using MAMALASampler

#=deltap=[2.5e-6, 0.003, 0.1, 0.005, 0.005, 2.5e-6, 0.003, 0.1, 0.005, 0.005] #a good size for steps to keep it in bounds

scale=eye(Float64,10)
for n in 1:length(deltap)
    scale[n,n]=1.0/deltap[n]
end

pmeans=zeros(Float64,length(deltap))
include("pilotRun.jl")=#
covTTV=readdlm("pilotCov.txt",',')
pmeans=readdlm("pilotMeans.txt",',')
pmeans=vec(pmeans)

covTTVhalf= chol(covTTV)
B=covTTVhalf #sigma^(1/2)

#scale=inv(covTTVhalf)

#include("pilotRun.jl")
#covTTVhalf= chol(covTTV)
#scale=inv(covTTVhalf)


include("TTVmodel.jl")
include("MCMCdiagnostics.jl")

p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)

#mcsampler=MALA(1e-8) #requires tiny step size ~1e-14

#sampler=HMC(0.1,10)
#sampler=AM(1.0,10) #what is first input?

p0= Dict(:p=>zguess)

mcrange= BasicMCRange(nsteps=50000,burnin=2500)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])

#MCtuner=AcceptanceRateMCTuner(0.6,verbose=true)
MCtuner=VanillaMCTuner(verbose=true)
#=MCtuner=MAMALAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=true)
)=#

numsteps=5
driftsteps= logspace(0.0,2.0,numsteps)
aclengths=Array{Float64}(numsteps, length(pinit))
ess_array=Array{Float64}(numsteps, length(pinit))
accrate=Vector{Float64}(numsteps)

for i in 1:numsteps
   mcsampler=MALA(driftsteps[i])
  #mcsampler=HMC(driftsteps[i],5)
  #mcsampler=SMMALA(driftsteps[i], H -> softabs(H, 1000.))
  #=mcsampler=MAMALA(
    update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, tot, 10.),
    transform=H -> softabs(H, 1000.),
    driftstep=driftsteps[i],
    minorscale=0.001,
    c=0.01
  )=#
  job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
  println("Job: ", i," , Step size: ", driftsteps[i])
  run(job)
  outval=output(job).value
  acc=output(job).diagnosticvalues
  accrate[i]=mean(acc)
  aclengths[i,:]=aclength(outval, threshold=0.1, maxit=20000, jump=5)
  ess_array[i,:]=ess(output(job))
end

writedlm("../outputs/aclength_SMMALA_transformedTTVFaster.txt", aclengths, ",")
writedlm("../outputs/ess_SMMALA_transformedTTVFaster.txt", ess_array, ",")
writedlm("../outputs/accept_SMMALA_transformedTTVFaster.txt", accrate, ",")
writedlm("../outputs/step_SMMALA_transformedTTVFaster.txt", driftsteps, ",")

maxac=Vector(numsteps)
for i in 1:numsteps
    if(any(isnan(aclengths[i,:])))
        maxac[i]=Inf
    elseif accrate[i]>0.97 || accrate[i]<0.03
        maxac[i]=Inf
    else
        maxac[i]=maximum(aclengths[i,:])
    end
end

minind=indmin(maxac)
minstep=driftsteps[minind]

using Plots
plotly()

acplot=scatter(driftsteps,aclengths,
  layout=10,
  xaxis=( :log10),
  title=["mu_b" "P_b" "ti_b" "k_b" "h_b" "mu_c" "P_c" "ti_c" "k_c" "h_c"],
  leg=false)

essplot=scatter(driftsteps,ess_array,
  layout=10,
  xaxis=( :log10),
  title=["mu_b" "P_b" "ti_b" "k_b" "h_b" "mu_c" "P_c" "ti_c" "k_c" "h_c"],
  leg=false)

acceptplot= scatter(driftsteps, accrate,
  xaxis=:log10,
  xlabel="Step size",
  ylabel="Net acceptance rate",
  leg=false)
