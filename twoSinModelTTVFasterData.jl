#fitting twosinharmonicmodel for TTVFaster
#define scale and pmean externally
using ForwardDiff
include("sinusoidFunctions.jl")

ptrue=readdlm("pf_bData.txt", ',')
trueDatab= readdlm("sinHarmonicFit_bData.txt", ',')
trueDatac= readdlm("sinHarmonicFit_cData.txt", ',')

pinit=vec(ptrue)
#pguess=[-0.0004,0.0005,-0.007,0.001,0.01107,0.001,0.003,-0.003,0.001, 0.001008]
errscale=[0.001,0.0001,0.0001,0.0001,0.01,0.001,0.0001,0.001,0.0001]
errs=(2*rand(9)-1).*errscale
pguess=pinit+errs

#pmeans=zeros(10)
#scale=eye(10)
invscale=inv(scale)

zinit=scale*(pinit-pmeans)
zguess=scale*(pguess-pmeans)

#=function isvalid2(z)
    param=invscale*z+pmeans
    if abs(param[6])>0.1 || abs(param[7])>0.1 || abs(param[8])>0.1 || abs(param[9])>0.1
        return false
    elseif param[10]<0.0 ||param[10]>0.005
        return false
    else
        return true
    end
end=#

#=function isvalidSSM{T<:Number}(p::Vector{T})
    if p[5]<0 || p[10] < 0
        return false
    end
    return true
end
=#
function plogtarget{T<:Number}(z::Vector{T})
    param=invscale*z+pmeans
    #if !isvalidSSM(param)
    #    return -Inf
    #end

    tb=trueDatab[:,1]
    pb=vcat(param[1:4],param[9])
    yb=sinharmonicmodel(tb,pb)

    tc=trueDatac[:,1]
    pc=param[5:9]
    yc=sinharmonicmodel(tc,pc)

    chisqb=0.0
    for j in 1:length(tb)
      chisqb+= (yb[j]-trueDatab[j,2])^2/trueDatab[j,3]^2
    end

    chisqc=0.0
    for k in 1:length(tc)
      chisqc+= (yc[k]-trueDatac[k,2])^2/trueDatac[k,3]^2
    end

    chisq= chisqb + chisqc

    return -chisq/2.0
end

gconfig=ForwardDiff.GradientConfig(zinit)
function pgradlogtarget{T<:Number}(z::Vector{T})
    param=invscale*z+pmeans
    gstore=ForwardDiff.gradient(plogtarget,z,gconfig)
    return gstore #don't need to divide by scale b/c gradient is in terms of z
end

hconfig=ForwardDiff.HessianConfig(zinit)
function ptensorlogtarget{T<:Number}(z::Vector{T})
  param=invscale*z+pmeans
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
