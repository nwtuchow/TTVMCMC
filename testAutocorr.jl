#testAutocorrelation

using Distributions,ForwardDiff, Klara


include("MCMCdiagnostics.jl")
include("sinusoidFunctions.jl")
#=
k1=5.0
phi1=4*pi/5

k2=3.5
phi2=5*pi/9

pinit=[k1*cos(phi1), k1*sin(phi1), 3*pi/7, k2*cos(phi2),k2*sin(phi2),pi/4]
pguess=[-3.0, 4.0, 3*pi/7+1e-6, -0.5, 3.0, pi/4-1e-6] #further away requires longer burnin
=#
pinit=[-3.5,4.3,1.0,0.8, 3*pi/7]
pguess=[-3.0,4.0,0.9, 1.2, 3*pi/7 + 1e-6]

trueData=simData(pinit,100,0.1) #fit maxima appear to move around

function plogtarget{T<:Number}(param::Vector{T})
  xarr=trueData[:,1]
  y=twosinmodel(xarr,param)

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

p = BasicContMuvParameter(:p, logtarget=plogtarget, gradlogtarget=pgradlogtarget)
model = likelihood_model(p, false)
p0= Dict(:p=>pguess)
mcrange= BasicMCRange(nsteps=50000, burnin=2500)
outopts= Dict{Symbol, Any}(:monitor=>[:value],:diagnostics =>[:accept])
minstep=0.00011787686347935878
sampler=HMC(minstep,5)
MCtuner=VanillaMCTuner(verbose=true)
#MCtuner=AcceptanceRateMCTuner(0.6,verbose=true)
#maybe this is only meant for tuning MALA
job=BasicMCJob(model,sampler,mcrange, p0, tuner=MCtuner, outopts=outopts)


testaclengths=Vector(5)
test_ess= Vector(5)
accrate=Vector{Float64}(5)

for j in 1:5
  reset(job, pguess)
  run(job)
  outval=output(job).value
  acc=output(job).diagnosticvalues
  accrate[j]=mean(acc)
  testaclengths[j]=aclength(outval, threshold=0.1, maxit=25000, jump=5)
  test_ess[j]=ess(output(job))
end

#= testaclengths gives without tuner
testaclengths=Any[Any[5496,4236,6546,4866,5101],Any[4086,2726,9621,10651,3711],Any[11381,10591,13101,12516,10851],Any[6091,2791,7611,8181,5511],Any[4141,2196,5071,7086,3856]]

test_ess=Any[[9.81237,13.1151,9.86325,10.6473,13.4171],[11.656,18.2927,5.49052,4.91347,15.9102],[5.57558,7.62145,3.32754,3.47705,5.85616],[9.00248,17.137,5.70728,6.36588,14.1289],[11.3799,21.8135,9.58959,8.0519,16.0979]]

accrate=[0.952653,0.950526,0.950253,0.951032,0.951895]

=#
