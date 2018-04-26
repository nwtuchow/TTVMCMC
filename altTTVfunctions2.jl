#Functions and gradients for fitting TTVs
include("modifiedTTVFasterFunctions.jl")

using ForwardDiff, Distributions

use_logs = false #true not tested yet
verbose = true
const jmax0 = 5


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
function simDataset!{T<:Number}(p::Vector{T}, bdata::Array{T,2},cdata::Array{T,2};noise::T=5.0,jmax=jmax0)
    num_b=Int64(length(bdata)/3)
    num_c=Int64(length(cdata)/3)

    timeb=p[3]+p[2]*collect(0:(num_b-1))#linear ephemerus
    timec=p[8]+p[7]*collect(0:(num_c-1))

    alpha0 = abs(p[2]/p[7])^(2//3)
    b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)
    # Set up planets planar-planet types for the inner and outer planets:
    planet1=TTVFaster.Planet_plane_hk(p[1],p[2],p[3],p[4],p[5])
    planet2=TTVFaster.Planet_plane_hk(p[6],p[7],p[8],p[9],p[10])

    dtb=Vector{T}(num_b)
    dtc=Vector{T}(num_c)
    f1=Array(T,jmax+2,5)
    f2=Array(T,jmax+2,5)
    b=Array(T,jmax+2,3)

    compute_inner_ttv!(jmax, planet1, planet2, timeb, dtb, f1,f2, b, alpha0, b0)
    compute_outer_ttv!(jmax, planet1, planet2, timec, dtc, f1,f2, b, alpha0, b0)

    #add noise

    sigma=noise #minutes
    sigma=sigma/60/24 #days
    if noise!=0.0
        dist=Normal(0.0,sigma)
        errb= rand(dist,num_b)
        errc= rand(dist,num_c)

        dtb+=errb
        dtc+=errc
    end

    tb= timeb + dtb
    tc= timec + dtc

    #note when sigma=0 chisq gives NaN
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

  #rowe et al ephemeruses
  # timeb= tb-dtb
  #timec= tc-dtc
  tnumb=round(Int64, (tb-tb[1])/p[2] +1)
  tnumc=round(Int64, (tc-tc[1])/p[7] +1)

  #my linear ephemeruses
  timeb= p[3]+ p[2]*(tnumb-1)
  timec= p[8]+ p[7]*(tnumc-1)

  alpha0 = abs(p[2]/p[7])^(2//3) #should these be calculated outside instead?
  b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)

  # Set up planets planar-planet types for the inner and outer planets:
  planet1=TTVFaster.Planet_plane_hk(p[1],p[2],p[3],p[4],p[5])
  planet2=TTVFaster.Planet_plane_hk(p[6],p[7],p[8],p[9],p[10])

  ttvb=Array(T2,length(tb))
  ttvc=Array(T2,length(tc))
  # Define arrays to hold the TTV coefficients and Laplace coefficients:
  f1=Array(T2,jmax+2,5)
  f2=Array(T2,jmax+2,5)
  b=Array(T2,jmax+2,3)

  compute_inner_ttv!(jmax, planet1, planet2, timeb, ttvb, f1,f2, b, alpha0, b0)
  compute_outer_ttv!(jmax, planet1, planet2, timec, ttvc, f1,f2, b, alpha0, b0)

  chisqc=chisquared(tb,timeb+ttvb,eb)
  chisqb=chisquared(tc,timec+ttvc,ec)
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

  tnumb=round(Int64, (tb-tb[1])/p[2] +1)
  tnumc=round(Int64, (tc-tc[1])/p[7] +1)

  alpha0 = abs(p[2]/p[7])^(2//3)
  b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)

  function fb{Tp<:Number}(p::Vector{Tp})  #gives tb array
      #tnumb=round(Int64, (tb-tb[1])/p[2] +1)
      #my ephemeruses
      tlinb= p[3]+ p[2]*(tnumb-1)

      #alpha0 = abs(p[2]/p[7])^(2//3)
      #b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)

      planet1=TTVFaster.Planet_plane_hk(p[1],p[2],p[3],p[4],p[5])
      planet2=TTVFaster.Planet_plane_hk(p[6],p[7],p[8],p[9],p[10])
      altttvb=Array(Tp,length(tb))
      # alternative arrays as workspace
      altf1=Array(Tp,jmax+2,5)
      altf2=Array(Tp,jmax+2,5)
      altb=Array(Tp,jmax+2,3)
      #println("\naltb: ", typeof(altb))
      compute_inner_ttv!(jmax,planet1,planet2,tlinb,altttvb,altf1,altf2,altb,alpha0,b0)
      return tlinb+altttvb
  end


  function fc{Tp<:Number}(p::Vector{Tp})
      #tnumc=round(Int64, (tc-tc[1])/p[7] +1)
      tlinc= p[8]+ p[7]*(tnumc-1)

     # alpha0 = abs(p[2]/p[7])^(2//3)
      #b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)

      planet1=TTVFaster.Planet_plane_hk(p[1],p[2],p[3],p[4],p[5])
      planet2=TTVFaster.Planet_plane_hk(p[6],p[7],p[8],p[9],p[10])
      altttvc=Array(Tp,length(tc))
      # alternative arrays as workspace
      altf1=Array(Tp,jmax+2,5)
      altf2=Array(Tp,jmax+2,5)
      altb=Array(Tp,jmax+2,3)
      compute_outer_ttv!(jmax,planet1,planet2,tlinc,altttvc,altf1,altf2,altb,alpha0,b0)
      return tlinc+altttvc
  end


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



#########################################################################################
#hessian of a vector valued function with respect to p
function vhess{T<:Number}(fttv,x::Vector{T},tnum::Vector{Int64}, planet::Char)
    l=length(x)
    vhstore=Array{T}(l,l,0)
    hstore=Array{T}(l,l)
    lt=length(tnum)
    hcfg=ForwardDiff.HessianConfig{2}(x)#ForwardDiff v0.2 syntax
    for i in 1:lt
        function f(z)
            single=fttv(z,[tnum[i]],planet)
            return single[1]
        end
        ForwardDiff.hessian!(hstore,f,x,hcfg)
        vhstore=cat(3,vhstore,hstore)
    end
    return vhstore
end
#=
function vhess2{T<:Number}(fsingle,x::Vector{T},times::Vector{T}, planet::Char)
    l=length(x)
    lt=length(times)
    vhstore=Array{T}(lt,lt,0)
    hstore=Array{T}(lt,lt)

    hcfg=ForwardDiff.HessianConfig{2}(times)#ForwardDiff v0.2 syntax
    for i in 1:lt
        function f(t)
            single=fsingle(x,[t],planet)
            return single[1]
        end
        ForwardDiff.hessian!(hstore,f,times[i],hcfg)
        vhstore=cat(3,vhstore,hstore)
    end
    return vhstore
end=#



#hstore needs to be size n x n where n is length of parameter array
#takes around a minute on first run. Second run is 0.334565 seconds, 89.993 MB 5.91% gc time.
function hesstransit!{T<:Number}(p::Vector{T}, hstore::Array{T,2}, bdata::Array{T,2}, cdata::Array{T,2}, jconfig; jmax=jmax0)
  tb=bdata[:,1]
  tc=cdata[:,1]
  dtb=bdata[:,2]
  dtc=cdata[:,2]
  eb=bdata[:,3]
  ec=cdata[:,3]

  tnumb=round(Int64, (tb-tb[1])/p[2] +1)
  tnumc=round(Int64, (tc-tc[1])/p[7] +1)
#=
  #my ephemeruses
  timeb= p[3]+ p[2]*(tnumb-1)
  timec= p[8]+ p[7]*(tnumc-1)

  dtimeb= zeros(length(timeb),10)
  dtimeb[:,2]= (tnumb-1)
  dtimeb[:,3]= ones(Float64,length(timeb))

  dtimec= zeros(length(timec),10)
  dtimec[:,7]=(tnumc-1)
  dtimec[:,8]=ones(Float64,length(timec))

  #second derivatives of linear ephemeruses are all zero, so omit them
=#
  alpha0 = abs(p[2]/p[7])^(2//3) #should these be calculated outside instead?
  b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)

function fb{Tp<:Number}(p::Vector{Tp})  #gives tb array
    #tnumb=round(Int64, (tb-tb[1])/p[2] +1)
    #my ephemeruses
    tlinb= p[3]+ p[2]*(tnumb-1)

    #alpha0 = abs(p[2]/p[7])^(2//3)
    #b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)

    planet1=TTVFaster.Planet_plane_hk(p[1],p[2],p[3],p[4],p[5])
    planet2=TTVFaster.Planet_plane_hk(p[6],p[7],p[8],p[9],p[10])
    altttvb=Array(Tp,length(tb))
    # alternative arrays as workspace
    altf1=Array(Tp,jmax+2,5)
    altf2=Array(Tp,jmax+2,5)
    altb=Array(Tp,jmax+2,3)

    compute_inner_ttv!(jmax,planet1,planet2,tlinb,altttvb,altf1,altf2,altb,alpha0,b0)
    return tlinb+altttvb
end

function fc{Tp<:Number}(p::Vector{Tp})
    #tnumc=round(Int64, (tc-tc[1])/p[7] +1)
    tlinc= p[8]+ p[7]*(tnumc-1)

    #alpha0 = abs(p[2]/p[7])^(2//3)
    #b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)

    planet1=TTVFaster.Planet_plane_hk(p[1],p[2],p[3],p[4],p[5])
    planet2=TTVFaster.Planet_plane_hk(p[6],p[7],p[8],p[9],p[10])
    altttvc=Array(Tp,length(tc))
    # alternative arrays as workspace
    altf1=Array(Tp,jmax+2,5)
    altf2=Array(Tp,jmax+2,5)
    altb=Array(Tp,jmax+2,3)
    compute_outer_ttv!(jmax,planet1,planet2,tlinc,altttvc,altf1,altf2,altb,alpha0,b0)
    return tlinc+altttvc
end

function fttv{Tp<:Number}(p::Vector{Tp}, tnum::Vector{Int64},planet::Char)
    #alpha0 = abs(p[2]/p[7])^(2//3)
    #b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)

    planet1=TTVFaster.Planet_plane_hk(p[1],p[2],p[3],p[4],p[5])
    planet2=TTVFaster.Planet_plane_hk(p[6],p[7],p[8],p[9],p[10])

    # alternative arrays as workspace
    altf1=Array(Tp,jmax+2,5)
    altf2=Array(Tp,jmax+2,5)
    altb=Array(Tp,jmax+2,3)

    if (planet=='b' || planet=='B' ||planet=='1')
        tlin=p[3]+ p[2]*(tnum-1)
        ttvarr= Array(Tp,length(tlin))
        compute_inner_ttv!(jmax,planet1,planet2,tlin,ttvarr,altf1,altf2,altb,alpha0,b0)
        return (tlin+ttvarr)
    elseif (planet=='c' || planet=='C' ||planet=='2')
        tlin=p[8]+ p[7]*(tnum-1)
        ttvarr= Array(Tp,length(tlin))
        compute_outer_ttv!(jmax,planet1,planet2,tlin,ttvarr,altf1,altf2,altb,alpha0,b0)
        return (tlin+ttvarr)
    else
        println("Planet is neither inner nor outer. \n")
    end
end
  #=function fsingle{Tp<:Number}(p::Vector{Tp},time::Vector{T}, planet::Char)
    @assert length(time)==1
    planet1=TTVFaster.Planet_plane_hk(p[1],p[2],p[3],p[4],p[5])
    planet2=TTVFaster.Planet_plane_hk(p[6],p[7],p[8],p[9],p[10])
    singlettv=Array(Tp,length(time))
    # alternative arrays as workspace
    altf1=Array(Tp,jmax+2,5)
    altf2=Array(Tp,jmax+2,5)
    altb=Array(Tp,jmax+2,3)
    if (planet=='b' || planet=='B' ||planet=='1')
        compute_inner_ttv!(jmax,planet1,planet2,time,singlettv,altf1,altf2,altb,alpha0,b0)
    elseif (planet=='c' || planet=='C' ||planet=='2')
        compute_outer_ttv!(jmax,planet1,planet2,time,singlettv,altf1,altf2,altb,alpha0,b0)
    else
        println("Planet is neither inner nor outer. \n")
    end

    return singlettv #returns size 1 array
  end
=#
  tau_b=fb(p)
  tau_c=fc(p)

  dfb(x)=ForwardDiff.jacobian(fb,x,jconfig)
  dfc(x)=ForwardDiff.jacobian(fc,x,jconfig)
  dTTVbdp=dfb(p) #jacobians of TTV outputs
  dTTVcdp=dfc(p)

  #dTTVbdt=ForwardDiff.jacobian(fb2,timeb)
  #dTTVcdt=ForwardDiff.jacobian(fc2,timec)

  #dTTVbdp+= dTTVbdt*dtimeb
  #dTTVcdp+= dTTVcdt*dtimec

  #hessians of vector valued function
  d2TTVb=vhess(fttv,p,tnumb,'b')
  d2TTVc=vhess(fttv,p,tnumc,'c')

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

#=function hessian_finite_diff{T<:Real}(f::Function, x::Vector{T}; delta::T=1e-4)
  result = Array(T,(length(x),length(x)))
  for i in 1:length(x), j in 1:i
     result[i,j] = result[j,i] = hessian_finite_diff(f,x,i,j,delta_i=delta,delta_j=delta)
   end
  return result
end=#

function hessian_finite_diff{T<:Real}(f::Function, x::Vector{T}; delta::T=1e-6, scale::Vector{T} = ones(length(x)))
  result = Array(T,(length(x),length(x)))
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
        elseif any(isnan(A)) || any(isinf(A))
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
