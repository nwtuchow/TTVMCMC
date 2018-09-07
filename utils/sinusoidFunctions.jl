#sine model Functions
#takes B and pmeans externally
using Distributions

if !isdefined(:B) || !isdefined(:pmeans)
    B=eye(12)
    pmeans=zeros(12)
end

function to_z{T<:Number}(p::Vector{T})
    z= B \ (p-pmeans)
    return z
end

function to_p{T<:Number}(z::Vector{T})
    p=B*z+pmeans
    return p
end

function sinharmonicmodel{T1<:Number,T2<:Number}(x::Vector{T1},p::Vector{T2})
  return p[1]*sin.(p[5]*x) + p[2]*cos.(p[5]*x) + p[3]*sin.(2.0*p[5]*x) +p[4]*cos.(2.0*p[5]*x)
end

#use linear ephemeruses
function linsinharmonic{T1<:Number,T2<:Number}(tlin::Vector{T1}, p::Vector{T2})

    f=sinharmonicmodel(tlin,p)
    ttot= tlin + f
    return ttot
end


function linsinharmonicb{T1<:Int,T2<:Number}(tnumb::Vector{T1}, p::Vector{T2})
    tlinb= p[1]+(tnumb-1)*p[2]
    PTTV=1/(5/p[8] - 4/p[2])
    part1=p[3:6]
    pb=push!(part1,2*pi/PTTV)
    yb=linsinharmonic(tlinb,pb)
    return yb
end

function linsinharmonicc{T1<:Int,T2<:Number}(tnumc::Vector{T1}, p::Vector{T2})
    tlinc= p[7]+(tnumc-1)*p[8]
    PTTV=1/(5/p[8] - 4/p[2])
    part2=p[9:12]
    pc=push!(part2,2*pi/PTTV)
    yc=linsinharmonic(tlinc,pc)
    return yc
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
