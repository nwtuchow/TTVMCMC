#Testing MCMC in Klara
using Klara
using MAMALASampler

covTTV=readdlm("pilotCov3.txt",',')
pmeans=readdlm("pilotMeans3.txt",',')
pmeans=vec(pmeans)

pstart=readdlm("pilotLast3.txt",',')
pstart=vec(pstart)
#=
diagcov=eye(Float64,10,10) #using only diagonals of cov matrix
for n in 1:10
    diagcov[n,n]=covTTV[n,n]
end
=#
covTTVhalf= ctranspose(chol(covTTV))
B=covTTVhalf #sigma^(1/2)

include("TTVmodel3old.jl")
include("MCMCdiagnostics.jl")

p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)

zstart=to_z(pstart)
p0= Dict(:p=>zstart)
nstep=5000
mcrange= BasicMCRange(nsteps=nstep)
#mcrange= BasicMCRange(nsteps=20000)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])

#MCtuner=AcceptanceRateMCTuner(0.6,verbose=true)
MCtuner=VanillaMCTuner(verbose=true)
#=MCtuner=MAMALAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=true)
)=#

numsteps=10
driftsteps= logspace(-1.0,0.3,numsteps)
aclengths=Array{Float64}(numsteps, length(pinit))
ess_array=Array{Float64}(numsteps, length(pinit))
accrate=Vector{Float64}(numsteps)

for i in 1:numsteps
  #mcsampler=MALA(driftsteps[i])
  mcsampler=HMC(driftsteps[i],7)
  #mcsampler=SMMALA(driftsteps[i], H -> simple_posdef(H, a=1500.))
  #=mcsampler=MAMALA(
    update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, tot, 10.),
    transform=H -> simple_posdef(H, a=1500.),
    driftstep=driftsteps[i],
    minorscale=0.01,
    c=0.01
  )=#
  job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
  println("Job: ", i," , Step size: ", driftsteps[i])
  run(job)
  outval=output(job).value
  acc=output(job).diagnosticvalues
  accrate[i]=mean(acc)
  println("Net Acceptance: ", 100.0*accrate[i],  "\%")
  aclengths[i,:]=aclength(outval, threshold=0.1, maxit=nstep, jump=1,useabs=true)
  ess_array[i,:]=nstep./aclengths[i,:]
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
    if(any(isnan(aclengths[i,:])) || accrate[i]<0.1 ||accrate[i]>0.9)
        maxac[i]=Inf
        miness[i]=0.0
    else
        maxac[i]=maximum(aclengths[i,:])
        miness[i]=minimum(ess_array[i,:])
    end
end

minind=indmin(maxac)
minstep=driftsteps[minind]

maxind=indmax(miness)
maxstep=driftsteps[maxind]

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
