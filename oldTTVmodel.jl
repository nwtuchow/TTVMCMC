#TTVFaster statistical model
#now define scale externally as matrix
#takes B = sigma^(1/2) externally
#takes pmeans externally

#using ForwardDiff
include("altTTVfunctions2.jl")

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
pguess=[mu_b+2.0e-6,P_b,ti_b,k_b+0.005,h_b+0.002,mu_c-1e-6,P_c,ti_c,k_c-0.004,h_c+0.001]
#=
pguess=[mu_b+5.0e-6*rand()-2.5e-6,
  P_b,
  ti_b,
  k_b+0.01*rand()-0.005,
  h_b+0.01*rand()-0.005,
  mu_c+5e-6*rand()-2.5e-6,
  P_c,
  ti_c,
  k_c+0.01*rand()-0.005,
  h_c+0.01*rand()-0.005]
=#

#deltap=[2.5e-6, 0.003, 0.1, 0.005, 0.005, 2.5e-6, 0.003, 0.1, 0.005, 0.005] #a good size for steps to keep it in bounds
#bData=Array(Float64,200,3)
#cData=Array(Float64,150,3)
#simDataset!(pinit,bData,cData)
#writedlm("TTVFasterbData.txt",bData,",")
#writedlm("TTVFastercData.txt",cData,",")
bData=readdlm("TTVFasterbData.txt",',')
cData=readdlm("TTVFastercData.txt",',')

#TTVbRange=maximum(bData[:,2])-minimum(bData[:,2])
#TTVcRange=maximum(cData[:,2])-minimum(cData[:,2])

np=length(pinit)

jconfig=ForwardDiff.JacobianConfig(pinit)

# z:scaled parameter array, z=scale*p

#zinit=scale*(pinit-pmeans)
#zguess=scale*(pguess-pmeans)

#invscale=inv(scale)
zinit=B \ (pinit-pmeans)
zguess=B \ (pguess-pmeans)

function ploglikelihood(z::Vector{Float64})
  #param=invscale*z+pmeans
  param=B*z+pmeans
  if !isvalidTTV(param)
    return -Inf
  end
  chisq= fittransit(bData,cData,param)
  if isnan(chisq)
      return -Inf
  end
  return -chisq/2.0 #also need 2pi term, error term
end

function plogprior(z::Vector{Float64})
  #param=invscale*z+pmeans
  param=B*z+pmeans
  logprior=0.0

  if !isvalidTTV(param)
    return -Inf
  end

  tblogprior=0.0
  #eta=1e-6 #changed to 1e-9
  eta=1e-3
  b=1/(pinit[2]+sqrt(pi*eta))
  tblogprior+=log(b)
  if (param[3]<= ti_b + pinit[2]/2.0) && (param[3]>= ti_b - pinit[2]/2.0)
    tblogprior+=0.0
  elseif (param[3]> ti_b + pinit[2]/2.0)
    d2=param[3]-(ti_b + pinit[2]/2.0)
    tblogprior+= -d2^2 /eta
  elseif (param[3] < ti_b - pinit[2]/2.0)
    d1= param[3]-(ti_b - pinit[2]/2.0)
    tblogprior+= -d1^2/eta
  end

  tclogprior=0.0
  b2=1/(pinit[7]+sqrt(pi*eta))
  tclogprior+=log(b2)
  if (param[8]<= ti_c + pinit[7]/2.0) && (param[8]>= ti_c - pinit[7]/2.0)
    tclogprior+=0.0
  elseif (param[8]> ti_c + pinit[7]/2.0)
    d2=param[8]-(ti_c + pinit[7]/2.0)
    tclogprior+= -d2^2 /eta
  elseif (param[8] < ti_c - pinit[7]/2.0)
    d1= param[8]-(ti_c - pinit[7]/2.0)
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
    tot=plogprior(z) + ploglikelihood(z) #plus constant
   # println("\np: ", z)
   #println("\nlogtarget: ",tot)
    return tot
end

function pgradlogtarget(z::Vector{Float64})
  #param=invscale*z+pmeans
  param=B*z+pmeans
  gstore=Vector{eltype(z)}(length(z)) #store gradient
  gradlogprior=zeros(length(z))
  if !isvalidTTV(param)
      return zeros(Float64,10) #might get stuck here
  end

  eta=1e-3
  #t_b gradlogprior
  if (param[3]<= ti_b + pinit[2]/2.0) && (param[3]>= ti_b - pinit[2]/2.0)
    gradlogprior[3]+=0.0
  elseif (param[3]> ti_b + pinit[2]/2.0)
    d2=param[3]-(ti_b + pinit[2]/2.0)
    gradlogprior[3]+= -2*d2/eta
  elseif (param[3] < ti_b - pinit[2]/2.0)
    d1= param[3]-(ti_b - pinit[2]/2.0)
    gradlogprior+= -2*d1/eta
  end

  #t_c gradlogprior
  if (param[8]<= ti_c + pinit[7]/2.0) && (param[8]>= ti_c - pinit[7]/2.0)
    gradlogprior[8]+=0.0
  elseif (param[8]> ti_c + pinit[7]/2.0)
    d2=param[8]-(ti_c + pinit[7]/2.0)
    gradlogprior[8]+= -2*d2 /eta
  elseif (param[8] < ti_c - pinit[7]/2.0)
    d1= param[8]-(ti_c - pinit[7]/2.0)
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


  gradtransit!(param,gstore,bData,cData,jconfig)
  gradloglikelihood= -0.5*gstore
  for i in gradloglikelihood
    if isnan(i)
      return zeros(Float64,10)
    end
  end
  #println("\ngradlogtarget: ",gradlogprior+gradloglikelihood)
  #return invscale'*(gradlogprior+gradloglikelihood)
  gtot= (gradlogprior+gradloglikelihood)'*B
  return gtot[1,:]
end

function ptensorlogtarget(z::Vector{Float64})
  #param=invscale*z+pmeans
  param=B*z+pmeans
  l=length(z)
  if !isvalidTTV(param)
      return zeros(Float64,10,10) #might get stuck here
  end
  tensorlogprior=zeros(eltype(z),l,l)

  eta=1e-3

  #t_b tensorlogprior
  if (param[3]<= ti_b + pinit[2]/2.0) && (param[3]>= ti_b - pinit[2]/2.0)
    tensorlogprior[3,3]+=0.0
  elseif (param[3]> ti_b + pinit[2]/2.0)
    tensorlogprior[3,3]+= -2/eta
  elseif (param[3] < ti_b - pinit[2]/2.0)
    tensorlogprior[3,3]+= -2/eta
  end

  #t_c tensorlogprior
  if (param[8]<= ti_c + pinit[7]/2.0) && (param[8]>= ti_c - pinit[7]/2.0)
    tensorlogprior[8,8]+=0.0
  elseif (param[8]> ti_c + pinit[7]/2.0)
    tensorlogprior[8,8]+= -2/eta
  elseif (param[8] < ti_c - pinit[7]/2.0)
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


  hstore=Array{eltype(z)}(l,l) #store hessian
  hesstransit!(param,hstore,bData,cData,jconfig)

  tensorloglikelihood=-0.5*hstore
  for i in 1:l
    for j in 1:l
      #hstore[i,j]= hstore[i,j]/scale[i]/scale[j]
      if isnan(hstore[i,j])
        return zeros(Float64,10,10)
      end
    end
  end

  tottensor=tensorloglikelihood+tensorlogprior
  #return (invscale'*tottensor)*invscale
  tot=(B'*tottensor)*B
  return tot
end

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
