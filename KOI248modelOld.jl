#julia 5 compatible
#TTVFaster statistical model
#now define scale externally as matrix
#takes B = sigma^(1/2) externally (Lower diagonal)
#takes pmeans externally

#using ForwardDiff
include("TTVfunctions3old.jl")

pinit=readdlm("KOI248ptrue.txt",',')
pinit=vec(pinit)

pguess=copy(pinit)
pguess[1]=3.0e-5
pguess[4]=0.04
pguess[5]=-0.02
pguess[6]=2.0e-5
pguess[9]=0.03
pguess[10]=-0.008

bData=readdlm("KOI248bData.txt",',')
cData=readdlm("KOI248cData.txt",',')

np=length(pinit)

# z:transformed parameter array
zinit=to_z(pinit)
zguess=to_z(pguess)

thinit=to_theta(pinit)
thguess=to_theta(pguess)

jconfig=ForwardDiff.JacobianConfig(pinit)
jconfigz=ForwardDiff.JacobianConfig(zinit)
jconfigth=ForwardDiff.JacobianConfig(thinit)
hcfgz=ForwardDiff.HessianConfig{2}(zinit)
hcfgth=ForwardDiff.HessianConfig{2}(thinit)

function ploglikelihood(z::Vector{Float64})
  param=to_p(z)
  if !isvalidTTV(param)
    return -Inf
  end

  chisq=fittransitz(bData,cData,z)
  if isnan(chisq)
      return -Inf
  end
  return -chisq/2.0 #also need 2pi term, error term
end

function ploglikelihood_th(theta::Vector{Float64})
  param=from_theta(theta)
  if !isvalidTTV(param)
    return -Inf
  end

  chisq=fittransit_th(bData,cData,theta)
  if isnan(chisq)
      return -Inf
  end
  return -chisq/2.0 #also need 2pi term, error term
end


function plogprior(z::Vector{Float64})
  param=to_p(z)
  logprior=0.0

  if !isvalidTTV(param)
    return -Inf
  end

  tblogprior=0.0
  #eta=1e-6 #changed to 1e-9
  eta=1e-3
  b=1/(pinit[2]+sqrt(pi*eta))
  tblogprior+=log(b)
  if (param[3]<= pinit[3] + pinit[2]/2.0) && (param[3]>= pinit[3] - pinit[2]/2.0)
    tblogprior+=0.0
  elseif (param[3]> pinit[3] + pinit[2]/2.0)
    d2=param[3]-(pinit[3] + pinit[2]/2.0)
    tblogprior+= -d2^2 /eta
  elseif (param[3] < pinit[3] - pinit[2]/2.0)
    d1= param[3]-(pinit[3] - pinit[2]/2.0)
    tblogprior+= -d1^2/eta
  end

  tclogprior=0.0
  b2=1/(pinit[7]+sqrt(pi*eta))
  tclogprior+=log(b2)
  if (param[8]<= pinit[8] + pinit[7]/2.0) && (param[8]>= pinit[8] - pinit[7]/2.0)
    tclogprior+=0.0
  elseif (param[8]> pinit[8] + pinit[7]/2.0)
    d2=param[8]-(pinit[8] + pinit[7]/2.0)
    tclogprior+= -d2^2 /eta
  elseif (param[8] < pinit[8] - pinit[7]/2.0)
    d1= param[8]-(pinit[8] - pinit[7]/2.0)
    tclogprior+= -d1^2/eta
  end

  sigmae=0.1
  #mean =[0.0,0.0]
  ecov=(sigmae^2)*eye(2)
  evec_b= [param[4],param[5]]
  evecbprior= -0.5*(logdet(ecov) +2.0*log(2*pi) + evec_b' * inv(ecov) *evec_b)

  evec_c= [param[9],param[10]]
  eveccprior= -0.5*(logdet(ecov) +2.0*log(2*pi) + evec_c' * inv(ecov) *evec_c) #1 element array, need convert to scalar


  logprior=logprior+tblogprior+tclogprior +evecbprior[1]+eveccprior[1]
  return logprior
end

function plogprior_th(theta::Vector{Float64})
  #param=B*z+pmeans
  param=from_theta(theta)
  logprior=0.0

  if !isvalidTTV(param)
    return -Inf
  end

  tblogprior=0.0
  #eta=1e-6 #changed to 1e-9
  eta=1e-3
  b=1/(pinit[2]+sqrt(pi*eta))
  tblogprior+=log(b)
  if (param[3]<= pinit[3] + pinit[2]/2.0) && (param[3]>= pinit[3] - pinit[2]/2.0)
    tblogprior+=0.0
  elseif (param[3]> pinit[3] + pinit[2]/2.0)
    d2=param[3]-(pinit[3] + pinit[2]/2.0)
    tblogprior+= -d2^2 /eta
  elseif (param[3] < pinit[3] - pinit[2]/2.0)
    d1= param[3]-(pinit[3] - pinit[2]/2.0)
    tblogprior+= -d1^2/eta
  end

  tclogprior=0.0
  b2=1/(pinit[7]+sqrt(pi*eta))
  tclogprior+=log(b2)
  if (param[8]<= pinit[8] + pinit[7]/2.0) && (param[8]>= pinit[8] - pinit[7]/2.0)
    tclogprior+=0.0
  elseif (param[8]> pinit[8] + pinit[7]/2.0)
    d2=param[8]-(pinit[8] + pinit[7]/2.0)
    tclogprior+= -d2^2 /eta
  elseif (param[8] < pinit[8] - pinit[7]/2.0)
    d1= param[8]-(pinit[8] - pinit[7]/2.0)
    tclogprior+= -d1^2/eta
  end

  sigmae=0.1
  #mean =[0.0,0.0]
  ecov=(sigmae^2)*eye(2)
  evec_b= [param[4],param[5]]
  evecbprior= -0.5*(logdet(ecov) +2.0*log(2*pi) + evec_b' * inv(ecov) *evec_b)

  evec_c= [param[9],param[10]]
  eveccprior= -0.5*(logdet(ecov) +2.0*log(2*pi) + evec_c' * inv(ecov) *evec_c) #1 element array, need convert to scalar


  logprior=logprior+tblogprior+tclogprior +evecbprior[1]+eveccprior[1]
  return logprior
end


function plogtarget(z::Vector{Float64})
    tot=plogprior(z) + ploglikelihood(z)
    #tot=ploglikelihood(z)
    return tot
end

function plogtarget_th(theta::Vector{Float64})
    tot=plogprior_th(theta) + ploglikelihood_th(theta)
    return tot
end

function pgradloglikelihood(z::Vector{Float64})
    param=to_p(z)
    gstore=Vector{eltype(z)}(length(z)) #store gradient
    if !isvalidTTV(param)
        return zeros(Float64,10) #might get stuck here
    end
    gradtransitz!(z,gstore,bData,cData,jconfigz)
    gradloglikelihood= -0.5*gstore
    for i in gradloglikelihood
      if isnan(i)
        return zeros(Float64,10)
      end
    end
    return gradloglikelihood
end

function pgradloglikelihood_th(theta::Vector{Float64})
    param=from_theta(theta)
    gstore=Vector{eltype(theta)}(length(theta)) #store gradient
    if !isvalidTTV(param)
        return zeros(Float64,length(theta)) #might get stuck here
    end
    gradtransit_th!(theta,gstore,bData,cData,jconfigth)
    gradloglikelihood= -0.5*gstore
    for i in gradloglikelihood
      if isnan(i)
        return zeros(Float64,length(theta))
      end
    end
    return gradloglikelihood
end

function pgradlogprior(z::Vector{Float64})
     param=to_p(z)
     gradlogprior=zeros(length(z))
     if !isvalidTTV(param)
         return zeros(Float64,10) #might get stuck here
     end

     eta=1e-3
     #t_b gradlogprior
     if (param[3]<= pinit[3] + pinit[2]/2.0) && (param[3]>= pinit[3] - pinit[2]/2.0)
       gradlogprior[3]+=0.0
     elseif (param[3]> pinit[3] + pinit[2]/2.0)
       d2=param[3]-(pinit[3] + pinit[2]/2.0)
       gradlogprior[3]+= -2*d2/eta
     elseif (param[3] < pinit[3] - pinit[2]/2.0)
       d1= param[3]-(pinit[3] - pinit[2]/2.0)
       gradlogprior[3]+= -2*d1/eta
     end

     #t_c gradlogprior
     if (param[8]<= pinit[8] + pinit[7]/2.0) && (param[8]>= pinit[8] - pinit[7]/2.0)
       gradlogprior[8]+=0.0
     elseif (param[8]> pinit[8] + pinit[7]/2.0)
       d2=param[8]-(pinit[8] + pinit[7]/2.0)
       gradlogprior[8]+= -2*d2 /eta
     elseif (param[8] < pinit[8] - pinit[7]/2.0)
       d1= param[8]-(pinit[8] - pinit[7]/2.0)
       gradlogprior[8]+= -2*d1/eta
     end

     #e_b vector gradlogprior
     sigmae=0.1
     ecov=(sigmae^2)*eye(2)
     evec_b= [param[4],param[5]]
     gradevecb= -inv(ecov) *evec_b

     gradlogprior[4]+=gradevecb[1]
     gradlogprior[5]+=gradevecb[2]

     #e_c
     evec_c= [param[9],param[10]]
     gradevecc= -inv(ecov) *evec_c
     gradlogprior[9]+=gradevecc[1]
     gradlogprior[10]+=gradevecc[2]

     gradlogprior= gradlogprior'*B #change to gradient wrt z
     gradlogprior=gradlogprior[1,:]

     return gradlogprior
 end

function pgradlogprior_th(theta::Vector{Float64})
      param=from_theta(theta)
      gradlogprior=zeros(length(theta))
      if !isvalidTTV(param)
          return zeros(Float64,length(theta)) #might get stuck here
      end

      #e_b vector gradlogprior
      sigmae=0.1
      ecov=(sigmae^2)*eye(2)
      evec_b= [param[4],param[5]]
      gradevecb= -inv(ecov) *evec_b

      gradlogprior[2]+=gradevecb[1]
      gradlogprior[3]+=gradevecb[2]

      #e_c
      evec_c= [param[9],param[10]]
      gradevecc= -inv(ecov) *evec_c
      gradlogprior[5]+=gradevecc[1]
      gradlogprior[6]+=gradevecc[2]

      #gradlogprior= gradlogprior'*B #change to gradient wrt z
      #gradlogprior=gradlogprior[1,:]

      return gradlogprior
  end


function pgradlogtarget(z::Vector{Float64})
    gprior=pgradlogprior(z)
    glikelihood=pgradloglikelihood(z)
    tot=gprior+glikelihood
    #tot=glikelihood
    return tot
end

function pgradlogtarget_th(theta::Vector{Float64})
    gprior=pgradlogprior_th(theta)
    glikelihood=pgradloglikelihood_th(theta)
    tot=gprior+glikelihood
    return tot
end

#=
function pgradlogtarget(z::Vector{Float64})
  #param=B*z+pmeans
  param=to_p(z)
  gstore=Vector{eltype(z)}(length(z)) #store gradient
  gradlogprior=zeros(length(z))
  if !isvalidTTV(param)
      return zeros(Float64,10) #might get stuck here
  end

  eta=1e-3
  #t_b gradlogprior
  if (param[3]<= pinit[3] + pinit[2]/2.0) && (param[3]>= pinit[3] - pinit[2]/2.0)
    gradlogprior[3]+=0.0
  elseif (param[3]> pinit[3] + pinit[2]/2.0)
    d2=param[3]-(pinit[3] + pinit[2]/2.0)
    gradlogprior[3]+= -2*d2/eta
  elseif (param[3] < pinit[3] - pinit[2]/2.0)
    d1= param[3]-(pinit[3] - pinit[2]/2.0)
    gradlogprior[3]+= -2*d1/eta
  end

  #t_c gradlogprior
  if (param[8]<= pinit[8] + pinit[7]/2.0) && (param[8]>= pinit[8] - pinit[7]/2.0)
    gradlogprior[8]+=0.0
  elseif (param[8]> pinit[8] + pinit[7]/2.0)
    d2=param[8]-(pinit[8] + pinit[7]/2.0)
    gradlogprior[8]+= -2*d2 /eta
  elseif (param[8] < pinit[8] - pinit[7]/2.0)
    d1= param[8]-(pinit[8] - pinit[7]/2.0)
    gradlogprior[8]+= -2*d1/eta
  end

  #e_b vector gradlogprior
  sigmae=0.1
  ecov=(sigmae^2)*eye(2)
  evec_b= [param[4],param[5]]
  gradevecb= -inv(ecov) *evec_b

  gradlogprior[4]+=gradevecb[1]
  gradlogprior[5]+=gradevecb[2]

  #e_c
  evec_c= [param[9],param[10]]
  gradevecc= -inv(ecov) *evec_c
  gradlogprior[9]+=gradevecc[1]
  gradlogprior[10]+=gradevecc[2]

  gradlogprior= gradlogprior'*B #change to gradient wrt z
  gradlogprior=gradlogprior[1,:]

  #gradtransit!(param,gstore,bData,cData,jconfig)
  gradtransitz!(z,gstore,bData,cData,jconfigz)
  gradloglikelihood= -0.5*gstore
  for i in gradloglikelihood
    if isnan(i)
      return zeros(Float64,10)
    end
  end

  gtot= gradlogprior+gradloglikelihood
  #gtot= (gradlogprior+gradloglikelihood)'*B
  return gtot
end
=#
function ptensorlogprior(z::Vector{Float64})
    param=to_p(z)
    l=length(z)
    if !isvalidTTV(param)
        return eye(Float64,10)
    end
    tensorlogprior=zeros(eltype(z),l,l)
    eta=1e-3

    #t_b tensorlogprior
    if (param[3]<= pinit[3] + pinit[2]/2.0) && (param[3]>= pinit[3] - pinit[2]/2.0)
      tensorlogprior[3,3]+=0.0
    elseif (param[3]> pinit[3] + pinit[2]/2.0)
      tensorlogprior[3,3]+= -2/eta
    elseif (param[3] < pinit[3] - pinit[2]/2.0)
      tensorlogprior[3,3]+= -2/eta
    end

    #t_c tensorlogprior
    if (param[8]<= pinit[8] + pinit[7]/2.0) && (param[8]>= pinit[8] - pinit[7]/2.0)
      tensorlogprior[8,8]+=0.0
    elseif (param[8]> pinit[8] + pinit[7]/2.0)
      tensorlogprior[8,8]+= -2/eta
    elseif (param[8] < pinit[8] - pinit[7]/2.0)
      tensorlogprior[8,8]+= -2/eta
    end

    #e_b vector tensorlogprior
    sigmae=0.1
    ecov=(sigmae^2)*eye(2)
    tensorevec=-inv(ecov)
    tensorlogprior[4,4]+=tensorevec[1,1]
    tensorlogprior[5,5]+=tensorevec[2,2]
    tensorlogprior[4,5]+=tensorevec[1,2]
    tensorlogprior[5,4]+=tensorevec[2,1]

    #e_c
    tensorlogprior[9,9]+=tensorevec[1,1]
    tensorlogprior[10,10]+=tensorevec[2,2]
    tensorlogprior[9,10]+=tensorevec[1,2]
    tensorlogprior[10,9]+=tensorevec[2,1]

    tensorlogprior=(B'*tensorlogprior)*B #derivative wrt z
    return tensorlogprior
end

function ptensorlogprior_th(theta::Vector{Float64})
    param=from_theta(theta)
    l=length(theta)
    if !isvalidTTV(param)
        return eye(Float64,l)
    end
    tensorlogprior=zeros(eltype(theta),l,l)

    #e_b vector tensorlogprior
    sigmae=0.1
    ecov=(sigmae^2)*eye(2)
    tensorevec=-inv(ecov)
    tensorlogprior[2,2]+=tensorevec[1,1]
    tensorlogprior[3,3]+=tensorevec[2,2]
    tensorlogprior[2,3]+=tensorevec[1,2]
    tensorlogprior[3,2]+=tensorevec[2,1]

    #e_c
    tensorlogprior[5,5]+=tensorevec[1,1]
    tensorlogprior[6,6]+=tensorevec[2,2]
    tensorlogprior[5,6]+=tensorevec[1,2]
    tensorlogprior[6,5]+=tensorevec[2,1]

    #tensorlogprior=(B'*tensorlogprior)*B #derivative wrt z
    return tensorlogprior
end

function ptensorloglikelihood(z::Vector{Float64})
    param=to_p(z)
    l=length(z)
    if !isvalidTTV(param)
        return eye(Float64,10)
    end

    hstore=Array{eltype(z)}(l,l) #store hessian
    #hesstransit!(param,hstore,bData,cData,jconfig)
    hesstransitz!(z,hstore,bData,cData,jconfigz, hcfg=hcfgz)
    if any(isnan.(hstore))
        return eye(Float64,10)
    end
    tensorloglikelihood=-0.5*hstore

    return tensorloglikelihood
end

function ptensorloglikelihood_th(theta::Vector{Float64})
    param=from_theta(theta)
    l=length(theta)
    if !isvalidTTV(param)
        return eye(Float64,l)
    end

    hstore=Array{eltype(theta)}(l,l) #store hessian
    #hesstransit!(param,hstore,bData,cData,jconfig)
    hesstransit_th!(theta,hstore,bData,cData,jconfigth, hcfg=hcfgth)
    if any(isnan.(hstore))
        return eye(Float64,l)
    end
    tensorloglikelihood=-0.5*hstore

    return tensorloglikelihood
end


function ptensorlogtarget(z::Vector{Float64})
    tprior=ptensorlogprior(z)
    tlikelihood=ptensorloglikelihood(z)
    tot= tprior+tlikelihood
    #tot=tprior
    return 0.5*(tot+tot')
end

function ptensorlogtarget_th(theta::Vector{Float64})
    tprior=ptensorlogprior_th(theta)
    tlikelihood=ptensorloglikelihood_th(theta)
    tot= tprior+tlikelihood
    #tot=tprior
    return 0.5*(tot+tot')
end


#=
function ptensorlogtarget(z::Vector{Float64})
  #param=B*z+pmeans
  param=to_p(z)
  l=length(z)
  if !isvalidTTV(param)
      return zeros(Float64,10,10) #might get stuck here
  end
  tensorlogprior=zeros(eltype(z),l,l)

  eta=1e-3

  #t_b tensorlogprior
  if (param[3]<= pinit[3] + pinit[2]/2.0) && (param[3]>= pinit[3] - pinit[2]/2.0)
    tensorlogprior[3,3]+=0.0
  elseif (param[3]> pinit[3] + pinit[2]/2.0)
    tensorlogprior[3,3]+= -2/eta
  elseif (param[3] < pinit[3] - pinit[2]/2.0)
    tensorlogprior[3,3]+= -2/eta
  end

  #t_c tensorlogprior
  if (param[8]<= pinit[8] + pinit[7]/2.0) && (param[8]>= pinit[8] - pinit[7]/2.0)
    tensorlogprior[8,8]+=0.0
  elseif (param[8]> pinit[8] + pinit[7]/2.0)
    tensorlogprior[8,8]+= -2/eta
  elseif (param[8] < pinit[8] - pinit[7]/2.0)
    tensorlogprior[8,8]+= -2/eta
  end

  #e_b vector tensorlogprior
  sigmae=0.1
  ecov=(sigmae^2)*eye(2)
  tensorevec=-inv(ecov)
  tensorlogprior[4,4]+=tensorevec[1,1]
  tensorlogprior[5,5]+=tensorevec[2,2]
  tensorlogprior[4,5]+=tensorevec[1,2]
  tensorlogprior[5,4]+=tensorevec[2,1]

  #e_c
  tensorlogprior[9,9]+=tensorevec[1,1]
  tensorlogprior[10,10]+=tensorevec[2,2]
  tensorlogprior[9,10]+=tensorevec[1,2]
  tensorlogprior[10,9]+=tensorevec[2,1]

  tensorlogprior=(B'*tensorlogprior)*B #derivative wrt z
  #tensorlogprior= 0.5*(tensorlogprior+tensorlogprior')

  hstore=Array{eltype(z)}(l,l) #store hessian
  #hesstransit!(param,hstore,bData,cData,jconfig)
  hesstransitz!(z,hstore,bData,cData,jconfigz, hcfg=hcfgz)
  tensorloglikelihood=-0.5*hstore
  for i in 1:l
    for j in 1:l
      #hstore[i,j]= hstore[i,j]/scale[i]/scale[j]
      if isnan(hstore[i,j])
        return zeros(Float64,10,10)
      end
    end
  end

  tot=tensorloglikelihood+tensorlogprior
  #return (invscale'*tottensor)*invscale
  #tot=(B'*tottensor)*B
  return 0.5*(tot+tot')
end
=#

#pdata[param,iteration]
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
