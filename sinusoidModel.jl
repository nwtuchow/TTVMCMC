#simple sinusoid model
using Distributions,Klara,ForwardDiff
using Plots
plotly()

include("MCMCdiagnostics.jl")
include("sinusoidFunctions.jl")
#= It seems that MCMC chains often converge to different values than the initial
values used. Could it be that the islands in the distribution of h and k  be almost
as good fits as the true values?=#

#=
k1=5.0
phi1=4*pi/5

k2=3.5
phi2=5*pi/9
=#
#pinit=[k1*cos(phi1), k1*sin(phi1), 3*pi/7, k2*cos(phi2),k2*sin(phi2),pi/4]
#pguess=[-3.0, 4.0, 3*pi/7+1e-6, -0.5, 3.0, pi/4-1e-6] #further away requires longer burnin
pinit=[-3.5,4.3,1.0,0.8, 3*pi/7]
pguess=[-3.0,4.0,0.9, 1.2, 3*pi/7 + 1e-6]
#trueData=simData(pinit,100,0.1) #fit maxima appear to move around
trueData=readdlm("sinharmonicData.txt", ',')

function plogtarget{T<:Number}(param::Vector{T})
  xarr=trueData[:,1]
  y=sinharmonicmodel(xarr,param)

  chisq=0.0
  for j in 1:length(xarr)
    chisq+= (y[j]-trueData[j,2])^2/trueData[j,3]^2
  end

  return -chisq/2.0
end


gconfig=ForwardDiff.GradientConfig(pinit)
function pgradlogtarget{T<:Number}(param::Vector{T})
    gstore=Vector{T}(length(param))
    ForwardDiff.gradient!(gstore,plogtarget,param,gconfig)
    return gstore
end

numsteps=15
driftsteps= logspace(-8.5,-6.0,numsteps)
#no proposals accepted for steps >~0.0003
aclengths=Array{Float64}(numsteps, length(pinit))
ess_array=Array{Float64}(numsteps, length(pinit))
accrate=Vector{Float64}(numsteps)
p = BasicContMuvParameter(:p, logtarget=plogtarget, gradlogtarget=pgradlogtarget)
model = likelihood_model(p, false)
p0= Dict(:p=>pguess)
mcrange= BasicMCRange(nsteps=50000, burnin=2500)
outopts= Dict{Symbol, Any}(:monitor=>[:value], :diagnostics=>[:accept])
#MCtuner=AcceptanceRateMCTuner(0.6,verbose=false)
MCtuner=VanillaMCTuner(verbose=true)


for i in 1:numsteps
  sampler=MALA(driftsteps[i])
  job=BasicMCJob(model,sampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
  println("Job: ", i," , Step size: ", driftsteps[i])
  run(job)
  outval=output(job).value
  acc=output(job).diagnosticvalues
  accrate[i]=mean(acc)
  aclengths[i,:]=aclength(outval, threshold=0.1, maxit=20000, jump=5)
  ess_array[i,:]=ess(output(job))
end

writedlm("../outputs/aclength_MALA_2sinharmonic.txt", aclengths, ",")
writedlm("../outputs/ess_MALA_2sinharmonic.txt", ess_array, ",")
writedlm("../outputs/accept_MALA_2sinharmonic.txt", accrate, ",")
writedlm("../outputs/step_MALA_2sinharmonic.txt", driftsteps, ",")


acplot=scatter(driftsteps,aclengths,
  layout=5,
  xaxis=( :log10),
  title=["p1" "p2" "p3" "p4" "p5"],
  leg=false)

essplot=scatter(driftsteps,ess_array,
  layout=5,
  xaxis=( :log10),
  title=["p1" "p2" "p3" "p4" "p5"],
  leg=false)

acceptplot= scatter(driftsteps, accrate,
  xaxis=:log10,
  xlabel="HMC step size",
  ylabel="Net acceptance rate",
  leg=false)

maxac=Vector(numsteps)
for i in 1:numsteps
    if(any(isnan(aclengths[i,:])))
        maxac[i]=Inf
    else
        maxac[i]=maximum(aclengths[i,:])
    end
end

minind=indmin(maxac)
minstep=driftsteps[minind]


#running chain for minstep

#=
MCtuner=AcceptanceRateMCTuner(0.6,verbose=true)
sampler=HMC(minstep,5)
testaclengths=Vector(5)
for j in 1:5
  job=BasicMCJob(model,sampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
  run(job)
  outval=output(job).value
  testaclengths[j]=aclength(outval, threshold=0.05, maxit=10000)
end

#using second run minstep testaclength gives nans
r1=collect(1:10:10000)
ac1=autocorr(r1,outval[1,:])
ac2=autocorr(r1,outval[2,:])
ac3=autocorr(r1,outval[3,:])
ac4=autocorr(r1,outval[4,:])
ac5=autocorr(r1,outval[5,:])
ac6=autocorr(r1,outval[6,:])

acplot=plot(r1,[ac1,ac2,ac3,ac4,ac5,ac6],
            label=["p1" "p2" "p3" "p4" "p5" "p6"],
            xlabel="Lag",
            ylabel="Autocorrelation function",
            title= "HMC($minstep, 5)")
=#
