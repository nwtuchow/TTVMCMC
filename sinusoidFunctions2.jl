#sine model Functions
#requires B and pmeans
using Distributions

function to_z{T<:Number}(p::Vector{T})
    z= B \ (p-pmeans)
    return z
end

function to_p{T<:Number}(z::Vector{T})
    p=B*z+pmeans
    return p
end

function sinmodel{T1<:Number,T2<:Number}(x::Vector{T1},p::Vector{T2})
  return p[1]*sin.(p[3]*x) +p[2]*cos.(p[3]*x)
end

function sinmodel4{T1<:Number,T2<:Number}(x::Vector{T1},p::Vector{T2})
    return p[1]*cos.(p[3]*x + p[2])
end

function sinharmonicmodel{T1<:Number,T2<:Number}(x::Vector{T1},p::Vector{T2})
  return p[1]*sin.(p[5]*x) + p[2]*cos.(p[5]*x) + p[3]*sin.(2.0*p[5]*x) +p[4]*cos.(2.0*p[5]*x)
end

function sinharmonicmodel3{T1<:Number,T2<:Number}(x::Vector{T1},p::Vector{T2})
    return p[1]*cos.(p[5]*x +p[2])+ p[3]*cos.(2*p[5]*x + p[4])
end

#use linear ephemeruses
function linsinharmonic{T1<:Number,T2<:Number}(tlin::Vector{T1}, p::Vector{T2})
    #=k1=sqrt(p[1]^2 + p[2]^2)
    t0=acos(p[1]/k1)/p[5]
    if sign(t0) == -1
        k1= -sqrt(p[1]^2 + p[2]^2)
        t0=acos(p[1]/k1)/p[5]
    end

    P=2*pi/p[5]
    tlin= t0 +(tnum-1)*P=#
    #sinhar in minutes tlin in days
    f=sinharmonicmodel(tlin,p)
    ttot= tlin + f
    return ttot
end

function linsinharmonic3{T1<:Number,T2<:Number}(tlin::Vector{T1}, p::Vector{T2})
    f=sinharmonicmodel3(tlin,p)
    ttot= tlin + f
    return ttot
end

function linsin4{T1<:Number,T2<:Number}(tlin::Vector{T1}, p::Vector{T2})
    f=sinmodel4(tlin,p)
    ttot = tlin +f
    return ttot
end

#timeb=p[1]+(tnumb-1)*p[2]+p[3]*sin(p[7]*x) + p[4]*cos(p[7]*x) + p[5]*sin(2.0*p[7]*x) +p[6]*cos(2.0*p[7]*x)
#timec=p[8]+(tnumb-1)*p[9]+p[10]*sin(p[14]*x) + p[11]*cos(p[14]*x) + p[12]*sin(2.0*p[14]*x) +p[13]*cos(2.0*p[14]*x)
#
function linsinharmonicb{T1<:Int,T2<:Number}(tnumb::Vector{T1}, p::Vector{T2})
    tlinb= p[1]+(tnumb-1)*p[2]
    #PTTV=1/(5/p[8] - 4/p[2])
    #part1=p[3:6]
    #pb=push!(part1,PTTV)
    pb=p[3:7]
    yb=linsinharmonic(tlinb,pb)
    return yb
end

function linsinharmonicc{T1<:Int,T2<:Number}(tnumc::Vector{T1}, p::Vector{T2})
    tlinc= p[8]+(tnumc-1)*p[9]
    #PTTV=1/(5/p[8] - 4/p[2])
    #part2=p[9:12]
    #pc=push!(part2,PTTV)
    pc=p[10:14]
    yc=linsinharmonic(tlinc,pc)
    return yc
end

function linsinharmonicb2{T1<:Int,T2<:Number}(tnumb::Vector{T1}, p::Vector{T2})
    tlinb= p[1]+(tnumb-1)*p[2]
    PTTV=1/(5/p[8] - 4/p[2])
    part1=p[3:6]
    pb=push!(part1,2*pi/PTTV)
    yb=linsinharmonic(tlinb,pb)
    return yb
end

function linsinharmonicc2{T1<:Int,T2<:Number}(tnumc::Vector{T1}, p::Vector{T2})
    tlinc= p[7]+(tnumc-1)*p[8]
    PTTV=1/(5/p[8] - 4/p[2])
    part2=p[9:12]
    pc=push!(part2,2*pi/PTTV)
    yc=linsinharmonic(tlinc,pc)
    return yc
end

function linsinharmonicb3{T1<:Int,T2<:Number}(tnumb::Vector{T1}, p::Vector{T2})
    tlinb= p[1]+(tnumb-1)*p[2]
    PTTV=1/(5/p[8] - 4/p[2])
    part1=p[3:6]
    pb=push!(part1,2*pi/PTTV)
    yb=linsinharmonic3(tlinb,pb)
    return yb
end

function linsinharmonicc3{T1<:Int,T2<:Number}(tnumc::Vector{T1}, p::Vector{T2})
    tlinc= p[7]+(tnumc-1)*p[8]
    PTTV=1/(5/p[8] - 4/p[2])
    part2=p[9:12]
    pc=push!(part2,2*pi/PTTV)
    yc=linsinharmonic3(tlinc,pc)
    return yc
end

function linsinb4{T1<:Int,T2<:Number}(tnumb::Vector{T1}, p::Vector{T2})
    tlinb= p[1]+(tnumb-1)*p[2]
    PTTV=1/(5/p[6] - 4/p[2])
    part1=p[3:4]
    pb=push!(part1,2*pi/PTTV)
    yb=linsin4(tlinb,pb)
    return yb
end

function linsinc4{T1<:Int,T2<:Number}(tnumc::Vector{T1}, p::Vector{T2})
    tlinc= p[5]+(tnumc-1)*p[6]
    PTTV=1/(5/p[6] - 4/p[2])
    part2=p[7:8]
    pc=push!(part2,2*pi/PTTV)
    yc=linsin4(tlinc,pc)
    return yc
end



function twosinharmonicmodel{T1<:Number,T2<:Number}(x::Vector{T1},p::Vector{T2})
  sin1=p[1]*sin.(p[5]*x) + p[2]*cos.(p[5]*x) + p[3]*sin.(2.0*p[5]*x) +p[4]*cos.(2.0*p[5]*x)
  sin2=p[6]*sin.(p[10]*x) + p[7]*cos.(p[10]*x) + p[8]*sin.(2.0*p[10]*x) +p[9]*cos.(2.0*p[10]*x)
  return sin1+sin2
end

function twosinharmonicmodel2{T1<:Number,T2<:Number}(x::Vector{T1},p::Vector{T2})
    sin1= p[1]*cos.(2*pi*x/p[5] + p[2]) + p[3]*cos.(4*pi*x/p[5] +p[4])
    sin2= p[6]*cos.(2*pi*x/p[10] + p[7]) + p[8]*cos.(4*pi*x/p[10] +p[9])
    return sin1+sin2
end

function doublesinmodel{T1<:Number,T2<:Number}(x::Vector{T1},p::Vector{T2})
  return p[1]*sin.(p[3]*x) + p[2]*cos.(p[3]*x) + p[4]*sin.(p[6]*x) +p[5]*cos.(p[6]*x)
end


function simData(p,len,err; start=0.0,stop=30.0)
  xarray=collect(linspace(start,stop,len))
  #yarray= sinharmonicmodel(xarray,p)
  yarray= twosinharmonicmodel(xarray,p,Pscale=1)
  dist=Normal(0.0, err)
  noise=rand(dist,len)
  yarray+=noise

  dataArr= Array{Float64}(len,3)
  for i in 1:len
    dataArr[i,1]=xarray[i]
    dataArr[i,2]=yarray[i]
    dataArr[i,3]=err
  end
  return dataArr
end

function simDatab(tnumb,p,err)
  len=length(tnumb)
  yarray= linsinharmonicb(tnumb,p)
  dist=Normal(0.0, err)
  noise=rand(dist,len)
  yarray+=noise

  dataArr= Array{Float64}(len,3)
  for i in 1:len
    dataArr[i,1]=tnumb[i]
    dataArr[i,2]=yarray[i]
    dataArr[i,3]=err
  end
  return dataArr
end

function simDatac(tnumc,p,err)
  len=length(tnumc)
  yarray= linsinharmonicc(tnumc,p)
  dist=Normal(0.0, err)
  noise=rand(dist,len)
  yarray+=noise

  dataArr= Array{Float64}(len,3)
  for i in 1:len
    dataArr[i,1]=tnumc[i]
    dataArr[i,2]=yarray[i]
    dataArr[i,3]=err
  end
  return dataArr
end

function simDatab2(tnumb,p,err)
  len=length(tnumb)
  yarray= linsinharmonicb2(tnumb,p)
  dist=Normal(0.0, err)
  noise=rand(dist,len)
  yarray+=noise

  dataArr= Array{Float64}(len,3)
  for i in 1:len
    dataArr[i,1]=tnumb[i]
    dataArr[i,2]=yarray[i]
    dataArr[i,3]=err
  end
  return dataArr
end

function simDatac2(tnumc,p,err)
  len=length(tnumc)
  yarray= linsinharmonicc2(tnumc,p)
  dist=Normal(0.0, err)
  noise=rand(dist,len)
  yarray+=noise

  dataArr= Array{Float64}(len,3)
  for i in 1:len
    dataArr[i,1]=tnumc[i]
    dataArr[i,2]=yarray[i]
    dataArr[i,3]=err
  end
  return dataArr
end

function simDatab3(tnumb,p,err)
  len=length(tnumb)
  yarray= linsinharmonicb3(tnumb,p)
  dist=Normal(0.0, err)
  noise=rand(dist,len)
  yarray+=noise

  dataArr= Array{Float64}(len,3)
  for i in 1:len
    dataArr[i,1]=tnumb[i]
    dataArr[i,2]=yarray[i]
    dataArr[i,3]=err
  end
  return dataArr
end

function simDatac3(tnumc,p,err)
  len=length(tnumc)
  yarray= linsinharmonicc3(tnumc,p)
  dist=Normal(0.0, err)
  noise=rand(dist,len)
  yarray+=noise

  dataArr= Array{Float64}(len,3)
  for i in 1:len
    dataArr[i,1]=tnumc[i]
    dataArr[i,2]=yarray[i]
    dataArr[i,3]=err
  end
  return dataArr
end

function simDatab4(tnumb,p,err)
  len=length(tnumb)
  yarray= linsinb4(tnumb,p)
  dist=Normal(0.0, err)
  noise=rand(dist,len)
  yarray+=noise

  dataArr= Array{Float64}(len,3)
  for i in 1:len
    dataArr[i,1]=tnumb[i]
    dataArr[i,2]=yarray[i]
    dataArr[i,3]=err
  end
  return dataArr
end

function simDatac4(tnumc,p,err)
  len=length(tnumc)
  yarray= linsinc4(tnumc,p)
  dist=Normal(0.0, err)
  noise=rand(dist,len)
  yarray+=noise

  dataArr= Array{Float64}(len,3)
  for i in 1:len
    dataArr[i,1]=tnumc[i]
    dataArr[i,2]=yarray[i]
    dataArr[i,3]=err
  end
  return dataArr
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
#=
function simDatab(p,len,err; start=0.0,stop=30.0)
  xarray=collect(linspace(start,stop,len))

  pb=vcat(p[1:4],p[9])
  yarray= sinharmonicmodel(xarray,pb)
  dist=Normal(0.0, err)
  noise=rand(dist,len)
  yarray+=noise

  dataArr= Array(Float64,len,3)
  for i in 1:len
    dataArr[i,1]=xarray[i]
    dataArr[i,2]=yarray[i]
    dataArr[i,3]=err
  end
  return dataArr
end

function simDatac(p,len,err; start=0.0,stop=30.0)
  xarray=collect(linspace(start,stop,len))

  pc=p[5:9]
  yarray= sinharmonicmodel(xarray,pc)
  dist=Normal(0.0, err)
  noise=rand(dist,len)
  yarray+=noise

  dataArr= Array(Float64,len,3)
  for i in 1:len
    dataArr[i,1]=xarray[i]
    dataArr[i,2]=yarray[i]
    dataArr[i,3]=err
  end
  return dataArr
end=#
