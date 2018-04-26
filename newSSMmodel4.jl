#changed SSM to only use single sinusoid for each TTV curve
#fitting SSM for TTVFaster
#define B and pmean externally
using ForwardDiff
include("sinusoidFunctions.jl")

ptrue=readdlm("pf_bData4.txt", ',')
trueDatab= readdlm("sinHarmonicFit_bData4.txt", ',')
trueDatac= readdlm("sinHarmonicFit_cData4.txt", ',')

pinit=vec(ptrue) #true values

P_b =10.5
P_c =13.0

P_TTV=1/(5/P_c-4/P_b)

#pguess=[pinit[1],P_b,0.003,5.7,pinit[5],P_c, 0.002,pi/2] #starting values
pguess=[pinit[1],P_b,0.007,5.7,pinit[5],P_c, 0.02,2.7] #starting values

#pmeans=zeros(8)
#scale=eye(8)

zinit=B \ (pinit-pmeans)
zguess=B \ (pguess-pmeans)

function isvalidSSM{T<:Number}(p::Vector{T})
    if p[2]<0 || p[6] < 0
        return false
    end
    return true
end

function plogtarget{T<:Number}(z::Vector{T})
    param=B*z+pmeans
    if !isvalidSSM(param)
        return -Inf
    end

    if param[4]<0.0
        param[4]+=2*pi
    elseif param[4]> 2*pi
        param[4]-=2*pi
    end

    if param[8]<0.0
        param[8]+=2*pi
    elseif param[8]> 2*pi
        param[8]-=2*pi
    end

    z= B \ (param-pmeans)

    tnumb=round(Int64,trueDatab[:,1])
    tb=trueDatab[:,2] #linear ephemeruses
    eb=trueDatab[:,3]

    yb= linsinb4(tnumb,param)

    tnumc=round(Int64,trueDatac[:,1])
    tc=trueDatac[:,2]
    ec=trueDatac[:,3]

    yc=linsinc4(tnumc,param)

    chisqb=0.0
    for j in 1:length(tb)
      chisqb+= (yb[j]-tb[j])^2/eb[j]^2
    end

    chisqc=0.0
    for k in 1:length(tc)
      chisqc+= (yc[k]-tc[k])^2/ec[k]^2
    end

    chisq= chisqb + chisqc

    return -chisq/2.0
end

gconfig=ForwardDiff.GradientConfig(zinit)
function pgradlogtarget{T<:Number}(z::Vector{T})
    #param=B*z+pmeans
    gstore=ForwardDiff.gradient(plogtarget,z,gconfig)
    return gstore #don't need to divide by scale b/c gradient is in terms of z
end

hconfig=ForwardDiff.HessianConfig(zinit)
function ptensorlogtarget{T<:Number}(z::Vector{T})
  #param=B*z+pmeans
  hstore=ForwardDiff.hessian(plogtarget,z,hconfig)
  return hstore
end

function samplecov(pdata::Array{Float64,2})
  pshape=size(pdata)
  np=pshape[1]
  ni=pshape[2]
  cov=zeros(Float64,np,np)
  means=Vector(np)
  for i in 1:np
    means[i]=mean(pdata[i,:])
  end

  for m in 1:ni
    cov+= (pdata[:,m]-means)*(pdata[:,m]-means)'
  end
  cov=cov/(ni-1)
  return cov
end
