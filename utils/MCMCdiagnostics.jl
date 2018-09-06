#diagnose MCMC chains

#autocorr
#gives autocorrelation function at for MCMC chain at an integer lag
#inputs
#   k: integer lag
#   chain: MCMC chain
#outputs
#   tot: autocorrelation function
function autocorr{T<:Number}(k::Int64,chain::Vector{T})
    n=length(chain)
    mu= mean(chain)
    sigma=std(chain)
    t=1
    sum1=0
    while t <= n-k
      sum1+=(chain[t]-mu)*(chain[t+k]-mu)
      t+=1
    end
    tot= sum1/(n-k)/(sigma^2)
    return tot
end

#variation of autocorr, takes multi-parameter chain instead
#chain first index: param, second index step number
#returns vector of autocorrelation function values
function autocorr{T<:Number}(k::Int64, chain::Array{T,2})
    dims= size(chain)
    l=dims[1]
    acorr=Vector{T}(l)
    for i in 1:l
      acorr[i]=autocorr(k,chain[i,:])
    end
    return acorr
end

#variation of autocorr, takes range of ints instead of single integer lag
function autocorr{T<:Number}(r1::Vector{Int64}, chain::Vector{T})
  ac=Vector{T}(length(r1))
  for i in 1:length(r1)
    ac[i] =autocorr(r1[i], chain)
  end
  return ac
end

#variation of autocorr takes range of ints  and multiparameter mcmc chain
#returns array of autocorrelation values
function autocorr{T<:Number}(r1::Vector{Int64},chain::Array{T,2})
    rlen=length(r1)
    dims=size(chain)
    l=dims[1]
    ac=Array{T}(l,rlen)
    for j in 1:l
        ac[j,:]=autocorr(r1,chain[j,:])
    end
    return ac
end

#aclength
#function to find the autocorrelation length of an mcmc chain
#i.e. when autocorr function drops below a certain threshold
#inputs
#   chain: mcmc chain
#   threshold: user defined threshold for min autocorr func
#   maxit: max lag to search for intercept at
#   jump: increase in lag each step
#   useabs: use absolute value of autocorr func?
#outputs
#   j: integer autocorrelation length
function aclength{T<:Number}(chain::Vector{T};threshold=0.05, maxit=1000, jump=1, useabs=false)
  j=1
  while j<maxit
    ac=autocorr(j,chain)
    if useabs #set as abs because sometimes get large negatives
        if (abs(ac) <=threshold) && abs(autocorr(j+1,chain) <=threshold)
            return j
        end
    else
        if ac <= threshold
          return j
        end
    end
    j+=jump
  end
  return NaN
end

#variation using multiparameter mcmc chain, outputing array of int autocorr lengths
function aclength{T<:Number}(chain::Array{T,2};threshold=0.05, maxit=1000,jump=1, useabs=false)
  dims= size(chain)
  l=dims[1]
  al=Vector(l)
  for k in 1:l
    al[k]=aclength(chain[k,:], threshold=threshold, maxit=maxit,jump=jump, useabs=useabs)
  end
  return al
end

#cornerUncertainty
#gives uncertainties for marginal posterior distribution
#inputs:
#   outval: mcmc chain
#   quantiles: selected quantiles
#outputs:
#   outarr: array of medians and upper and lower uncertainties
function cornerUncertainty{T<:Number}(outval::Array{T,2}, quantiles=[0.16, 0.5, 0.84])
    numparam=size(outval)[1]
    outarr=Array{T}(numparam,3)
    for q in 1:numparam
        qls=quantile(outval[q,:],quantiles)
        outarr[q,1]=qls[2]
        outarr[q,2]=qls[3]-qls[2]
        outarr[q,3]=qls[2]-qls[1]
    end
    #row 1: median 2:upper 3:lower
    return outarr
end
