#Testing MCMC in Klara
using Klara
using MAMALASampler

covTTV=readdlm("pilotCov3.txt",',')
pmeans=readdlm("pilotMeans3.txt",',')
pmeans=vec(pmeans)

pstart=readdlm("pilotLast3.txt",',')
pstart=vec(pstart)

diagcov=eye(Float64,10,10) #using only diagonals of cov matrix
for n in 1:10
    diagcov[n,n]=covTTV[n,n]
end

covTTVhalf= chol(diagcov)
B=covTTVhalf #sigma^(1/2)

include("TTVmodel3.jl")
include("MCMCdiagnostics.jl")

p= BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model= likelihood_model(p, false)


zstart=to_z(pstart)
p0= Dict(:p=>zstart)

numsteps=5000000
#burnin=500000
mcrange= BasicMCRange(nsteps=numsteps, thinning=10)

outopts = Dict{Symbol, Any}(:monitor=>[:value],
  :diagnostics=>[:accept])

#MCtuner=AcceptanceRateMCTuner(0.6,verbose=true)
#MCtuner=VanillaMCTuner(verbose=true)
MCtuner=MAMALAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=true)
)

#minstep=0.000774263682681127
minstep=0.245
#mcsampler=MALA(minstep)
mcsampler=MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=minstep,
  minorscale=0.01,
  c=0.01
)


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

writedlm("../../../Documents/Exoplanet_ttv_data/values_transformedTTVFasterMAMALAminstep2.txt", outval, ",")
writedlm("../../../Documents/Exoplanet_ttv_data/accept_transformedTTVFasterMAMALAminstep2.txt", outacc, ",")

using PyPlot
using PyCall
using LaTeXStrings
@pyimport corner
corner.corner(outval', labels=[L"\mathbf{\mu_b}",L"\mathbf{P_b}",L"\mathbf{t_{i,b}}",L"\mathbf{k_b}",L"\mathbf{h_b}",L"\mathbf{\mu_c}",L"\mathbf{P_c}",L"\mathbf{t_{i,c}}",L"\mathbf{k_c}",L"\mathbf{h_c}"],
quantiles=[0.16, 0.5, 0.84],
show_titles=true)
