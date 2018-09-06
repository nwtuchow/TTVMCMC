#Functions and gradients for fitting TTVs
#requires B and pmeans
if !isdefined(:B) || !isdefined(:pmeans)
    B=eye(10)
    pmeans=zeros(10)
end

include("../../TTVFaster/Julia/compute_ttv.jl") #change to path of forked TTVFaster
using TTVFaster
using ForwardDiff, Distributions

use_logs = false #true not tested yet
verbose = true
const jmax0 = 5

function to_z{T<:Number}(p::Vector{T})
    z= B \ (p-pmeans)
    return z
end

function to_p{T<:Number}(z::Vector{T})
    p=B*z+pmeans
    return p
end


function chisquared{T1<:Number,T2<:Number}(obs::Vector{T1},calc::Vector{T2},err::Vector{T1})
    @assert(length(obs)==length(calc))
    chisq=0.0
    for i in 1:length(obs)
        chisq+=(obs[i]-calc[i])^2 /err[i]^2
    end
    return chisq
end

#gives boolean value as to whether p is in bounds
#changed name from isvalid to isvalidTTV
function isvalidTTV{T<:Number}(p::Vector{T})
  if p[1]<0.0 || p[6]<0.0 #assert positive mass ratios
    return false
  elseif p[1]>1.0 || p[6]>1.0 #planets must be smaller than star
    return false
  elseif p[2]<0.0 || p[7]<0.0 #postive periods
    return false
  elseif p[2]>p[7] #inner planet should have shorter period
    return false
  elseif (p[4]^2+p[5]^2) > 1.0 || (p[9]^2+p[10]^2) > 1.0 #eccentriciy between 0 and 1
    return false
  end
  return true
end

#simulate dataset given parameter array, num_b x 3 bData array,and num_c x 3 cData
function simDataset!{T<:Number}(p::Vector{T}, bdata::Array{T,2},cdata::Array{T,2};
    noiseb::T=5.0,noisec::T=5.0, jmax=jmax0,
    num_b=collect(1:(size(bdata)[1])),
    num_c=collect(1:(size(cdata)[1])))

    timeb=p[3]+p[2]*(num_b-1)#linear ephemerus
    timec=p[8]+p[7]*(num_c-1)

    alpha0 = abs(p[2]/p[7])^(2//3)
    b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)
    # Set up planets planar-planet types for the inner and outer planets:
    planet1=TTVFaster.Planet_plane_hk(p[1],p[2],p[3],p[4],p[5])
    planet2=TTVFaster.Planet_plane_hk(p[6],p[7],p[8],p[9],p[10])

    dtb=Vector{T}(length(num_b))
    dtc=Vector{T}(length(num_c))
    f1=Array{T}(jmax+2,5)
    f2=Array{T}(jmax+2,5)
    b=Array{T}(jmax+2,3)

    TTVFaster.compute_inner_ttv!(jmax, planet1, planet2, timeb, dtb, f1,f2, b, alpha0, b0)
    TTVFaster.compute_outer_ttv!(jmax, planet1, planet2, timec, dtc, f1,f2, b, alpha0, b0)

    #add noise

    sigmab=noiseb #minutes
    sigmab=sigmab/60/24 #days
    sigmac=noisec #minutes
    sigmac=sigmac/60/24 #days
    if noiseb!=0.0 && noisec!=0.0
        distb=Normal(0.0,sigmab)
        distc=Normal(0.0,sigmac)
        errb= rand(distb,length(num_b))
        errc= rand(distc,length(num_c))

        dtb+=errb
        dtc+=errc
    end

    tb= timeb + dtb
    tc= timec + dtc

    #note when sigma=0 chisq gives NaN
    for i in 1:length(num_b)
        bdata[i,1]=tb[i]
        bdata[i,2]=dtb[i]
        bdata[i,3]=sigmab
    end

    for j in 1:length(num_c)
        cdata[j,1]=tc[j]
        cdata[j,2]=dtc[j]
        cdata[j,3]=sigmac
    end
end




#planet b transit times
function ftimeb{T1<:Number}(p::Vector{T1}, tnumb::Vector{Int64},alpha0::Number,b0::Array; jmax=jmax0)
    tlinb= p[3]+ p[2]*(tnumb-1)

    planet1=TTVFaster.Planet_plane_hk(p[1],p[2],p[3],p[4],p[5])
    planet2=TTVFaster.Planet_plane_hk(p[6],p[7],p[8],p[9],p[10])
    altttvb=Array{T1}(length(tlinb))
    # alternative arrays as workspace
    altf1=Array{T1}(jmax+2,5)
    altf2=Array{T1}(jmax+2,5)
    altb=Array{T1}(jmax+2,3)

    TTVFaster.compute_inner_ttv!(jmax,planet1,planet2,tlinb,altttvb,altf1,altf2,altb,alpha0,b0)
    return tlinb+altttvb
end

#f(p(z))
function fptimeb{T1<:Number}(z::Vector{T1}, tnumb::Vector{Int64},alpha0::Number,b0::Array; jmax=jmax0)
    param=to_p(z)
    return ftimeb(param,tnumb,alpha0,b0,jmax=jmax)
end

#planet c transit times
function ftimec{T1<:Number}(p::Vector{T1}, tnumc::Vector{Int64},alpha0::Number,b0::Array; jmax=jmax0)
    tlinc= p[8]+ p[7]*(tnumc-1)

    planet1=TTVFaster.Planet_plane_hk(p[1],p[2],p[3],p[4],p[5])
    planet2=TTVFaster.Planet_plane_hk(p[6],p[7],p[8],p[9],p[10])
    altttvc=Array{T1}(length(tlinc))
    # alternative arrays as workspace
    altf1=Array{T1}(jmax+2,5)
    altf2=Array{T1}(jmax+2,5)
    altb=Array{T1}(jmax+2,3)
    TTVFaster.compute_outer_ttv!(jmax,planet1,planet2,tlinc,altttvc,altf1,altf2,altb,alpha0,b0)
    return tlinc+altttvc
end

#f(p(z))
function fptimec{T1<:Number}(z::Vector{T1}, tnumc::Vector{Int64},alpha0::Number,b0::Array; jmax=jmax0)
    param=to_p(z)
    return ftimec(param,tnumc,alpha0,b0,jmax=jmax)
end

#Gives combined chi squared value for both planets
#bData and cData are arrays with
#    row 1: times
#    row2: TTVs
#    row3: uncertainty
#p is paramter array
# [mu_b,P_b,ti_b,k_b,h_b,mu_c,P_c,ti_c,k_c,h_c]
#mu is planet-star mass ratio, P are periods(days),ti is first transit, k and h are ecc. vector
function fittransit{T<:Number,T2<:Number}(bdata::Array{T,2},cdata::Array{T,2},p::Vector{T2};jmax=jmax0)
  @assert(length(p)==10)

  tb=bdata[:,1] #from rowe et al 2014, measured transit time not linear ephemerus
  tc=cdata[:,1]
  dtb=bdata[:,2]
  dtc=cdata[:,2]
  eb=bdata[:,3]
  ec=cdata[:,3]

  tnumb=round.(Int64, (tb-p[3])/p[2] +1)
  tnumc=round.(Int64, (tc-p[8])/p[7] +1)

  alpha0 = abs(p[2]/p[7])^(2//3) #should these be calculated outside instead?
  b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)

  fb(param) = ftimeb(param,tnumb,alpha0,b0,jmax=jmax)
  fc(param) = ftimec(param,tnumc,alpha0,b0,jmax=jmax)

  tau_b=fb(p)
  tau_c=fc(p)
  chisqc=chisquared(tb,tau_b,eb)
  chisqb=chisquared(tc,tau_c,ec)
  return chisqb + chisqc
end

#same as fittransit but takes transformed parameters
function fittransitz{T<:Number,T2<:Number}(bdata::Array{T,2},cdata::Array{T,2},z::Vector{T2};jmax=jmax0)
  @assert(length(z)==10)

  tb=bdata[:,1] # measured transit time not linear ephemerus
  tc=cdata[:,1]
  dtb=bdata[:,2]
  dtc=cdata[:,2]
  eb=bdata[:,3]
  ec=cdata[:,3]
  p=to_p(z)
  tnumb=round.(Int64, (tb-p[3])/p[2] +1)
  tnumc=round.(Int64, (tc-p[8])/p[7] +1)

  alpha0 = abs(p[2]/p[7])^(2//3) #should these be calculated outside instead?
  b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)
  #note zram stands for z param
  fpb(zram) = fptimeb(zram,tnumb,alpha0,b0,jmax=jmax)
  fpc(zram) = fptimec(zram,tnumc,alpha0,b0,jmax=jmax)

  tau_b=fpb(z)
  tau_c=fpc(z)
  chisqc=chisquared(tb,tau_b,eb)
  chisqb=chisquared(tc,tau_c,ec)
  return chisqb + chisqc
end

######################################################################################
#gradient function
function gradtransit!{T<:Number,T2<:Number}(p::Vector{T},gstore::Vector{T},bdata::Array{T2,2},cdata::Array{T2,2},jconfig; jmax=jmax0)
  tb=bdata[:,1]
  tc=cdata[:,1]
  dtb=bdata[:,2]
  dtc=cdata[:,2]
  eb=bdata[:,3]
  ec=cdata[:,3]

  tnumb=round.(Int64, (tb-p[3])/p[2] +1)
  tnumc=round.(Int64, (tc-p[8])/p[7] +1)

  alpha0 = abs(p[2]/p[7])^(2//3)
  b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)

  fb(param) = ftimeb(param,tnumb,alpha0,b0,jmax=jmax)
  fc(param) = ftimec(param,tnumc,alpha0,b0,jmax=jmax)

  tau_b=fb(p)
  tau_c=fc(p)

  #ttv1 and 2 Array{Float64,1}
  dTTVbdp=ForwardDiff.jacobian(fb,p,jconfig) #derivatives of transit time array with respect to params
  dTTVcdp=ForwardDiff.jacobian(fc,p,jconfig)

  for j in 1:length(p)
    dchsqb=0.0
    dchsqc=0.0
    for i in 1:length(tb)
      dchsqb+= 2*(tau_b[i]-tb[i])*(dTTVbdp[i,j]) /eb[i]^2
    end

    for i2 in 1:length(tc)
      dchsqc+= 2*(tau_c[i2]-tc[i2])*(dTTVcdp[i2,j]) /ec[i2]^2
    end
    gstore[j]=(dchsqb+dchsqc)
  end
end

#gradient with respect to z
function gradtransitz!{T<:Number,T2<:Number}(z::Vector{T},gstore::Vector{T},bdata::Array{T2,2},cdata::Array{T2,2},jconfig; jmax=jmax0)
  tb=bdata[:,1]
  tc=cdata[:,1]
  dtb=bdata[:,2]
  dtc=cdata[:,2]
  eb=bdata[:,3]
  ec=cdata[:,3]

  p=to_p(z)
  tnumb=round.(Int64, (tb-p[3])/p[2] +1)
  tnumc=round.(Int64, (tc-p[8])/p[7] +1)

  alpha0 = abs(p[2]/p[7])^(2//3)
  b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)

  fpb(zram) = fptimeb(zram,tnumb,alpha0,b0,jmax=jmax)
  fpc(zram) = fptimec(zram,tnumc,alpha0,b0,jmax=jmax)

  tau_b=fpb(z)
  tau_c=fpc(z)

  #ttv1 and 2 Array{Float64,1}
  dTTVbdz=ForwardDiff.jacobian(fpb,z,jconfig) #derivatives of transit time array with respect to params
  dTTVcdz=ForwardDiff.jacobian(fpc,z,jconfig)

  for j in 1:length(p)
    dchsqb=0.0
    dchsqc=0.0
    for i in 1:length(tb)
      dchsqb+= 2*(tau_b[i]-tb[i])*(dTTVbdz[i,j]) /eb[i]^2
    end

    for i2 in 1:length(tc)
      dchsqc+= 2*(tau_c[i2]-tc[i2])*(dTTVcdz[i2,j]) /ec[i2]^2
    end
    gstore[j]=(dchsqb+dchsqc)
  end
end

#########################################################################################
#hessian of a vector valued function with respect to x
function vhess{T<:Number}(fttv,x::Vector{T},tnum::Vector{Int64}, planet::Char,hcfg)
    l=length(x)
    vhstore=Array{T}(l,l,0)
    hstore=Array{T}(l,l)
    lt=length(tnum)
    #hcfg=ForwardDiff.HessianConfig{2}(x)#ForwardDiff v0.2 syntax
    for i in 1:lt
        function f(xi)
            single=fttv(xi,[tnum[i]],planet)
            return single[1]
        end
        ForwardDiff.hessian!(hstore,f,x,hcfg)
        vhstore=cat(3,vhstore,hstore)
    end
    return vhstore
end

#hstore needs to be size n x n where n is length of parameter array
#takes around a minute on first run. Second run is 0.334565 seconds, 89.993 MB 5.91% gc time.
#maybe also add hconfig outside

function hesstransit!{T<:Number}(p::Vector{T}, hstore::Array{T,2}, bdata::Array{T,2}, cdata::Array{T,2}, jconfig; jmax=jmax0, hcfg=ForwardDiff.HessianConfig{2}(p))
  tb=bdata[:,1]
  tc=cdata[:,1]
  dtb=bdata[:,2]
  dtc=cdata[:,2]
  eb=bdata[:,3]
  ec=cdata[:,3]

  tnumb=round.(Int64, (tb-p[3])/p[2] +1)
  tnumc=round.(Int64, (tc-p[8])/p[7] +1)

  alpha0 = abs(p[2]/p[7])^(2//3) #should these be calculated outside instead?
  b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)

  fb(param) = ftimeb(param,tnumb,alpha0,b0,jmax=jmax)
  fc(param) = ftimec(param,tnumc,alpha0,b0,jmax=jmax)


function fttv{Tp<:Number}(p::Vector{Tp}, tnum::Vector{Int64},planet::Char)
    if (planet=='b' || planet=='B' ||planet=='1')
        return ftimeb(p,tnum,alpha0,b0,jmax=jmax)
    elseif (planet=='c' || planet=='C' ||planet=='2')
        return ftimec(p,tnum,alpha0,b0,jmax=jmax)
    else
        println("Planet is neither inner nor outer. \n")
        return NaN
    end
end

  tau_b=fb(p)
  tau_c=fc(p)

  dfb(x)=ForwardDiff.jacobian(fb,x,jconfig)
  dfc(x)=ForwardDiff.jacobian(fc,x,jconfig)
  dTTVbdp=dfb(p) #jacobians of TTV outputs
  dTTVcdp=dfc(p)

  #hessians of vector valued function
  d2TTVb=vhess(fttv,p,tnumb,'b',hcfg)
  d2TTVc=vhess(fttv,p,tnumc,'c',hcfg)

  #d2chisq/dpi/dpj= ∑ (2*(dTTV[k]/dpi)*(dTTV/dpj)/σ^2 + 2*(TTV-obs)*(d2TTV[k]/dpi/dpj)/σ^2)
  # see http://www.juliadiff.org/ForwardDiff.jl/advanced_usage.html#hessian-of-a-vector-valued-function
  for k in 1:length(p)
    for j in 1:length(p)
      d2chisqb=0.0
      d2chisqc=0.0
      for i in 1:length(tb)
          d2chisqb+= 2*(dTTVbdp[i,j])*(dTTVbdp[i,k])/eb[i]^2 +2*(tau_b[i]-tb[i])*d2TTVb[j,k,i]/eb[i]^2
      end

      for i2 in 1:length(tc)
          d2chisqc+= 2*(dTTVcdp[i2,j])*(dTTVcdp[i2,k])/ec[i2]^2 +2*(tau_c[i2]-tc[i2])*d2TTVc[j,k,i2]/ec[i2]^2
      end

      hstore[j,k]=d2chisqb+d2chisqc
    end
  end
end

#second dervatives with respect to z
function hesstransitz!{T<:Number}(z::Vector{T}, hstore::Array{T,2}, bdata::Array{T,2}, cdata::Array{T,2}, jconfig; jmax=jmax0, hcfg=ForwardDiff.HessianConfig{2}(z))
  tb=bdata[:,1]
  tc=cdata[:,1]
  dtb=bdata[:,2]
  dtc=cdata[:,2]
  eb=bdata[:,3]
  ec=cdata[:,3]

  p=to_p(z)
  tnumb=round.(Int64, (tb-p[3])/p[2] +1)
  tnumc=round.(Int64, (tc-p[8])/p[7] +1)

  alpha0 = abs(p[2]/p[7])^(2//3) #should these be calculated outside instead?
  b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)

  fpb(zram) = fptimeb(zram,tnumb,alpha0,b0,jmax=jmax)
  fpc(zram) = fptimec(zram,tnumc,alpha0,b0,jmax=jmax)


function fpttv{Tp<:Number}(z::Vector{Tp}, tnum::Vector{Int64},planet::Char)
    if (planet=='b' || planet=='B' ||planet=='1')
        return fptimeb(z,tnum,alpha0,b0,jmax=jmax)
    elseif (planet=='c' || planet=='C' ||planet=='2')
        return fptimec(z,tnum,alpha0,b0,jmax=jmax)
    else
        println("Planet is neither inner nor outer. \n")
        return NaN
    end
end

  tau_b=fpb(z)
  tau_c=fpc(z)

  dfpb(x)=ForwardDiff.jacobian(fpb,x,jconfig)
  dfpc(x)=ForwardDiff.jacobian(fpc,x,jconfig)
  dTTVbdz=dfpb(z) #jacobians of TTV outputs
  dTTVcdz=dfpc(z)

  #hessians of vector valued function
  d2TTVb=vhess(fpttv,z,tnumb,'b',hcfg)
  d2TTVc=vhess(fpttv,z,tnumc,'c',hcfg)

  #d2chisq/dpi/dpj= ∑ (2*(dTTV[k]/dpi)*(dTTV/dpj)/σ^2 + 2*(TTV-obs)*(d2TTV[k]/dpi/dpj)/σ^2)
  # see http://www.juliadiff.org/ForwardDiff.jl/advanced_usage.html#hessian-of-a-vector-valued-function
  for k in 1:length(p)
    for j in 1:length(p)
      d2chisqb=0.0
      d2chisqc=0.0
      for i in 1:length(tb)
          d2chisqb+= 2*(dTTVbdz[i,j])*(dTTVbdz[i,k])/eb[i]^2 +2*(tau_b[i]-tb[i])*d2TTVb[j,k,i]/eb[i]^2
      end

      for i2 in 1:length(tc)
          d2chisqc+= 2*(dTTVcdz[i2,j])*(dTTVcdz[i2,k])/ec[i2]^2 +2*(tau_c[i2]-tc[i2])*d2TTVc[j,k,i2]/ec[i2]^2
      end

      hstore[j,k]=d2chisqb+d2chisqc
    end
  end
end


#eric ford's test functions
function hessian_finite_diff{T<:Real}(f::Function, x::Vector{T}, i::Integer, j::Integer; delta_i::T=1e-6, delta_j::T=delta_i)
  @assert 1<=i<=length(x)
  @assert 1<=j<=length(x)
  sum = f(x)
  xtmp = copy(x)
  xtmp[i] += delta_i
  sum -= f(xtmp)
  xtmp = copy(x)
  xtmp[j] += delta_j
  sum -= f(xtmp)
  xtmp[i] += delta_i
  sum += f(xtmp)
  return sum/(delta_i*delta_j)
end

function hessian_finite_diff{T<:Real}(f::Function, x::Vector{T}; delta::T=1e-6, scale::Vector{T} = ones(length(x)))
  result = Array{T}(length(x),length(x))
  for i in 1:length(x), j in 1:i
     result[i,j] = result[j,i] = hessian_finite_diff(f,x,i,j,delta_i=delta*scale[i],delta_j=delta*scale[j])
   end
  return result
end

function simple_posdef(A::Array{Float64,2}; a::Float64 = 1000.0)
    A=softabs(A,a)
    A=0.5*(A+A')
    for i in 1:10
        if isposdef(A)
            return A
        elseif any(isnan.(A)) || any(isinf.(A))
            return zeros(size(A))
        else
            A=softabs(A,a)
            A=0.5*(A+A')
        end
    end
    println("Not posdef")
    return A
end

function make_matrix_pd(A::Array{Float64,2}; epsabs::Float64 = 0.0, epsfac::Float64 = 1.0e-6)
  @assert(size(A,1)==size(A,2))
  println("make_matrix_pd input: ", A)
  flush(STDOUT)
  B = 0.5*(A+A')
  itt = 1
  while !isposdef(B)
	eigvalB,eigvecB = eig(B)
        pos_eigval_idx = eigvalB.>0.0 #has trouble if eigenvalues are complex, only keep if real
    #needs at least 1 eigenvalue positive
    neweigval = (epsabs == 0.0) ? epsfac*minimum(abs.(eigvalB[pos_eigval_idx])) : epsabs
	eigvalB[!pos_eigval_idx] = neweigval
	B = eigvecB *diagm(eigvalB)*eigvecB'
	println(itt,": ",B)
        flush(STDOUT)
        #cholB = chol(B)
	itt +=1
	if itt>size(A,1)
	  error("There's a problem in make_matrix_pd.\n")
  	  break
	end
  end
  return B
end
