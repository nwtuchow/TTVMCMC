#make simulated dataset
include("modifiedTTVFasterFunctions.jl")
include("TTVfunctions.jl")

using Distributions, Klara
const jmax = 5

function simDataset!{T<:Number}(p::Vector{T}, bdata::Array{T,2},cdata::Array{T,2})
    num_b=Int64(length(bdata)/3)
    num_c=Int64(length(cdata)/3)

    timeb=p[3]+p[2]*collect(0:(num_b-1))#linear ephemerus
    timec=p[8]+p[7]*collect(0:(num_c-1))

    alpha0 = abs(p[2]/p[7])^(2//3)
    b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)
    # Set up planets planar-planet types for the inner and outer planets:
    planet1=TTVFaster.Planet_plane_hk(p[1],p[2],p[3],p[4],p[5])
    planet2=TTVFaster.Planet_plane_hk(p[6],p[7],p[8],p[9],p[10])

    dtb=zeros(num_b)
    dtc=zeros(num_c)
    f1=Array(T,jmax+2,5)
    f2=Array(T,jmax+2,5)
    b=Array(T,jmax+2,3)

    compute_inner_ttv!(jmax, planet1, planet2, timeb, dtb, f1,f2, b, alpha0, b0)
    compute_outer_ttv!(jmax, planet1, planet2, timec, dtc, f1,f2, b, alpha0, b0)

    #add noise
    sigma=5.0 #minutes
    sigma=sigma/60/24 #days
    noise=Normal(0.0,sigma)
    errb= rand(noise,num_b)
    errc= rand(noise,num_c)

    dtb+=errb
    dtc+=errc

    tb= timeb + dtb
    tc= timec + dtc

    for i in 1:num_b
        bdata[i,1]=tb[i]
        bdata[i,2]=dtb[i]
        bdata[i,3]=sigma
    end

    for j in 1:num_c
        cdata[j,1]=tc[j]
        cdata[j,2]=dtc[j]
        cdata[j,3]=sigma
    end
end

#kepler 307b params
P_b =10.5
P_c =13.0

t0_b=784.0
t0_c=785.0

k_b= 0.011
k_c= 0.004

h_b= -0.04
h_c= -0.029

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

mu_b=8.0 # Me/Msun
mu_c=4.0

mu_b=3.003e-6*mu_b
mu_c=3.003e-6*mu_c

ti_b=t0_b
ti_c=t0_c

pinit=[mu_b,P_b,ti_b,k_b,h_b,mu_c,P_c,ti_c,k_c,h_c]

#=pguess=[mu_b+5.0e-6*rand()-2.5e-6,
  P_b+0.2*rand()-0.1,
  ti_b+0.1,
  k_b+0.01*rand()-0.005,
  h_b+0.01*rand()-0.005,
  mu_c+5e-6*rand()-2.5e-6,
  P_c+0.2*rand()-0.1,ti_c+0.1,
  k_c+0.01*rand()-0.005,
  h_c+0.01*rand()-0.005]
=#
bData=Array(Float64,200,3)
cData=Array(Float64,150,3)

bData2=Array(Float64,200,3)
cData2=Array(Float64,150,3)


simDataset!(pinit,bData,cData)

np=length(pinit)
jconfig=ForwardDiff.JacobianConfig(pinit)
gstore=Vector{eltype(pinit)}(np) #store gradient

function ploglikelihood(z::Vector{Float64})
  if !isvalid(z)
    return -Inf
  end
  chisq= fittransit(bData,cData,z)
  if isnan(chisq)
      return -Inf
  end
  return -chisq/2.0 #also need 2pi term, error term
end

function plogprior(z::Vector{Float64})
  logprior=0.0
  if !isvalid(z)
    return -Inf
  end

  return logprior
end

function plogtarget(z::Vector{Float64})
    tot=plogprior(z) + ploglikelihood(z) #plus constant
   # println("\np: ", z)
   #println("\nlogtarget: ",tot)
    return tot
end

function pgradlogtarget(z::Vector{Float64})
  gradlogprior=0.0
  if !isvalid(z)
      return zeros(Float64,10) #might get stuck here
  end
  gradtest!(z,gstore,bData,cData,jconfig)
  gradloglikelihood= -0.5*gstore
  for i in gradloglikelihood
    if isnan(i)
      return zeros(Float64,10)
    end
  end
  #println("\ngradlogtarget: ",gradlogprior+gradloglikelihood)
  return gradlogprior+gradloglikelihood
end

p = BasicContMuvParameter(:p, logtarget=plogtarget, gradlogtarget=pgradlogtarget)

model = likelihood_model(p, false)

sampler=MALA(1e-4)

p0= Dict(:p=>pinit)

mcrange= BasicMCRange(nsteps=5000000, burnin=1000)

outopts= Dict{Symbol, Any}(:monitor=>[:value, :logtarget], :diagnostics=>[:accept])

MCtuner=AcceptanceRateMCTuner(0.6,verbose=true)
#MCtuner=VanillaMCTuner(verbose=true)

job=BasicMCJob(model,sampler,mcrange, p0, tuner=MCtuner, outopts=outopts)

run(job)#  AssertionError: Log-target not finite: initial value out of support

out=output(job)
parray=out.value

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

#=
using Plots
plotly()
plotb= scatter(bData2[:,1],1440*bData2[:,2],
        ylabel= "TTV (min)",
        yerror=1440*bData[:,3],
        markersize=3,
        markershape=:circle,
        label="Planet b")

plotc= scatter(cData2[:,1],1440*cData2[:,2],
        xlabel= "Transit Time",
        ylabel= "TTV (min)",
        yerror=1440*cData[:,3],
        markersize=3,
        markershape=:circle,
        label="Planet c")

doubleplot=plot(plotb,plotc, layout=(2,1))
=#
