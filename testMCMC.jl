#Testing MCMC in Klara
using Klara

include("TTVfunctions.jl")

bData=readdlm("../koi0248.01.tt")
cData=readdlm("../koi0248.02.tt")

#try for Kepler 49, KOI 248
#using Jontof hutter 2016 estimates for params
P_b =7.2040
P_c =10.9123

t0_b=780.4529
t0_c=790.3470

k_b= 0.011
k_c= 0.006

h_b= 0.037
h_c= 0.027 #real

#adjust argument of periastron for TTVFaster
e_b=sqrt(k_b^2 +h_b^2)
peri_b= atan2(h_b,k_b)
peri_b-=pi/2

k_b=e_b*cos(peri_b)
h_b=e_b*sin(peri_b)

e_c=sqrt(k_c^2 +h_c^2)
peri_c= atan2(h_c,k_c)
peri_c-=pi/2

k_c=e_c*cos(peri_c)
h_c=e_c*sin(peri_c)

mu_b=9.65 # Me/Msun
mu_c=6.28

mu_b=3.003e-6*mu_b
mu_c=3.003e-6*mu_c

ti_b=bData[1,1]
ti_c=cData[1,1]

pinit=[mu_b,P_b,ti_b,k_b,h_b,mu_c,P_c,ti_c,k_c,h_c]

np=length(pinit)
jconfig=ForwardDiff.JacobianConfig(pinit)
gstore=Vector{eltype(pinit)}(np) #store gradient

function ploglikelihood(z::Vector{Float64})
  if !isvalid(z)
    return -Inf
  end
  chisq= fittransit(bData,cData,z)
  return -chisq/2.0 #also need 2pi term, error term
end

function plogprior(z::Vector{Float64})
  logprior=0.0
  if !isvalid(z)
    return -Inf
  end

  return logprior
end

function pgradlogtarget(z::Vector{Float64})
  gradlogprior=0.0
  if !isvalid(z)
      return 0.0 #might get stuck here
  end
  gradtest!(z,gstore,bData,cData,jconfig)
  gradloglikelihood= -0.5*gstore
  #println("\ngradlogtarget: ",gradlogprior+gradloglikelihood)
  return gradlogprior+gradloglikelihood
end

function plogtarget(z::Vector{Float64})
    tot=plogprior(z) + ploglikelihood(z) #plus constant
   # println("\np: ", z)
   #println("\nlogtarget: ",tot)
    return tot
end
p= BasicContMuvParameter(:p, logtarget=plogtarget,gradlogtarget=pgradlogtarget)

model= likelihood_model(p, false)

sampler=MALA(1e-4) #starts accepting after step ~600
#with MALA(0.1) get
#p: [54148.1,772.231,76.3912,60.5548,-478.733,51897.1,-366.328,62.9291,-73.625,589.833]
#logtarget: -Inf
#on first proposal because of large grad log target
#gradlogtarget: [1.08296e6,15295.9,190.99,1201.06,-9563.96,1.03795e6,-7544.31,-142.355,-1486.63,11795.0]
#way out of bounds


#sampler=HMC(0.1,10)
#sampler=AM(1.0,10) #what is first input?

p0= Dict(:p=>pinit)

mcrange= BasicMCRange(nsteps=5000000,burnin=1000)

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget],
  :diagnostics=>[:accept],
  :destination=>:iostream)

MCtuner=AcceptanceRateMCTuner(0.6,verbose=true)
#MCtuner=VanillaMCTuner(verbose=true)

job=BasicMCJob(model,sampler,mcrange, p0, tuner=MCtuner, outopts=outopts)

run(job)

out=output(job)


#=parray=out.value

using Plots
plotly()
massplot=histogram2d(parray[1,:]/3.003e-6,parray[6,:]/3.003e-6,
            xlabel="Planet b mass ratio",
            ylabel="Planet c mass ratio")

kplot=histogram2d(parray[4,:],parray[9,:],
            xlabel="k_b",
            ylabel="k_c")

hplot=histogram2d(parray[5,:],parray[10,:],
            xlabel="h_b",
            ylabel="h_c")

plot(massplot,kplot,hplot,layout=(1,3))
=#
