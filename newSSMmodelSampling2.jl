#using 2 sin Harmonic model on data from ttv faster optimization
using Klara
using MAMALASampler
ndim=12

include("MCMCdiagnostics.jl")

#=
pmeans=zeros(ndim)
B=eye(ndim)
include("newSSMmodel.jl")


fmat= -ptensorlogtarget(zinit) #estimate of fisher information matrix
covguess= fmat \ eye(14)
#covguess=inv(fmat)
covguess= 0.5*(covguess+covguess')
B=chol(covguess)

include("pilotRunSSM.jl")
=#
covSSM=readdlm("pilotCovSSM2.txt",',')
pmeans=readdlm("pilotMeansSSM2.txt",',')
pmeans=vec(pmeans)

B=chol(covSSM)
include("newSSMmodel2.jl")

p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)

p0= Dict(:p=>zguess)

numsteps=50000
burnin=2500
mcrange= BasicMCRange(nsteps=numsteps,burnin=burnin, thinning=1)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])


numsteps=15
driftsteps= logspace(-2.0,0.7,numsteps)
aclengths=Array{Float64}(numsteps, length(pinit))
ess_array=Array{Float64}(numsteps, length(pinit))
accrate=Vector{Float64}(numsteps)

#MCtuner=VanillaMCTuner(verbose=true)
MCtuner=MAMALAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=true)
)


for i in 1:numsteps
    #mcsampler=HMC(driftsteps[i],5)
    #mcsampler=MALA(driftsteps[i])
    mcsampler=MAMALA(
        update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, 50000, 10.),
        transform=H -> softabs(H, 1000.),
        driftstep=driftsteps[i],
        minorscale=0.001,
        c=0.01)
    #mcsampler=SMMALA(driftsteps[i], H -> softabs(H,1000.0))

    job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
    println("Job: ", i," , Step size: ", driftsteps[i])
    run(job)
    out=output(job)
    outval=out.value
    outacc=out.diagnosticvalues
    accrate[i]=mean(outacc)
    aclengths[i,:]=aclength(outval, threshold=0.1, maxit=20000, jump=5)
    ess_array[i,:]=ess(output(job))
end

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

acplot=Plots.scatter(driftsteps,aclengths,
  layout=12,
  xaxis=( :log10),
  title=["1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12"],
  leg=false)

essplot=Plots.scatter(driftsteps,ess_array,
  layout=12,
  xaxis=( :log10),
  title=["1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12"],
  leg=false)

acceptplot= Plots.scatter(driftsteps, accrate,
  xaxis=:log10,
  xlabel="Step size",
  ylabel="Net acceptance rate",
  leg=false)
