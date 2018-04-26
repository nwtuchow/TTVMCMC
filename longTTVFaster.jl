#Testing MCMC in Klara
using Klara
using MAMALASampler

#=deltap=[2.5e-6, 0.003, 0.1, 0.005, 0.005, 2.5e-6, 0.003, 0.1, 0.005, 0.005] #a good size for steps to keep it in bounds
#scale=Vector{Float64}(deltap)
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

numsteps=15000000
burnin=500000 #long burnin for MALA
mcrange= BasicMCRange(nsteps=numsteps,burnin=burnin, thinning=10)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])

#MCtuner=AcceptanceRateMCTuner(0.6,verbose=true)
MCtuner=VanillaMCTuner(verbose=true)
#=MCtuner=MAMALAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=true)
)=#

#minstep=0.000774263682681127
minstep=4.641588833612778
mcsampler=MALA(minstep)
#=mcsampler=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, div(tot,100), 10.),
  transform=H -> softabs(H, 1000.),
  driftstep=minstep,
  minorscale=0.001,
  c=0.01
)=#


job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
run(job)
out=output(job)
outval=out.value
outacc=out.diagnosticvalues

#=for i in 1:np
  for j in 1:(size(outval)[2])
    outval[i,j]=outval[i,j]./scale[i] #rescale to get parameter values
  end
end=#
for j in 1:(size(outval)[2])
  outval[:,j]=B*outval[:,j]+pmeans
end

writedlm("../../../Documents/Exoplanet_ttv_data/values_transformedTTVFasterMALAminstep4.txt", outval, ",")
writedlm("../../../Documents/Exoplanet_ttv_data/accept_transformedTTVFasterMALAminstep4.txt", outacc, ",")

using PyPlot
using PyCall
using LaTeXStrings
@pyimport corner
corner.corner(outval', labels=[L"\mathbf{\mu_b}",L"\mathbf{P_b}",L"\mathbf{t_{i,b}}",L"\mathbf{k_b}",L"\mathbf{h_b}",L"\mathbf{\mu_c}",L"\mathbf{P_c}",L"\mathbf{t_{i,c}}",L"\mathbf{k_c}",L"\mathbf{h_c}"],
quantiles=[0.16, 0.5, 0.84],
show_titles=true)
