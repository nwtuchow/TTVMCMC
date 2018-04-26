#changed SSM on TTVFaster to only use 12 parameters
#fitting twosinharmonicmodel for TTVFaster
#define B and pmean externally
using ForwardDiff
include("sinusoidFunctions2.jl")

ptrue=readdlm("pf_bData2.txt", ',')
trueDatab= readdlm("sinHarmonicFit_bData2.txt", ',')
trueDatac= readdlm("sinHarmonicFit_cData2.txt", ',')

pinit=vec(ptrue) #true values

#errscale=[0.001,0.0001,0.0001,0.0001,0.01,0.001,0.0001,0.001,0.0001]
#errs=(2*rand(9)-1).*errscale

#pguess=pinit+errs

P_b =10.5
P_c =13.0

P_TTV=1/(5/P_c-4/P_b)


pguess=[ptrue[1],P_b,0.003,0.005, 0.00003,0.0001, ptrue[7],P_c,-0.008,-0.01,-0.0001,0.0001] #starting values

#zinit=B \ (pinit-pmeans)
zinit=to_z(pinit)
#zguess=B \ (pguess-pmeans)
zguess=to_z(pguess)

function isvalidSSM{T<:Number}(p::Vector{T})
    if p[2]<0 || p[8] < 0
        return false
    end
    return true
end

function plogtarget{T<:Number}(z::Vector{T})
    #param=B*z+pmeans
    param=to_p(z)
    if !isvalidSSM(param)
        return -Inf
    end

    tnumb=round(Int64,trueDatab[:,1])
    tb=trueDatab[:,2] #linear ephemeruses
    eb=trueDatab[:,3]

    yb= linsinharmonicb2(tnumb,param)

    tnumc=round(Int64,trueDatac[:,1])
    tc=trueDatac[:,2]
    ec=trueDatac[:,3]

    yc=linsinharmonicc2(tnumc,param)

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
