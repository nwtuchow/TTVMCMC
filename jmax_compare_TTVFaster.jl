#compare TTVFaster calculations using different jmax
B=eye(10)
pmeans=zeros(10)
include("TTVmodel.jl")

function calcTTV{T<:Number}(p::Vector{T},tnumb::Array{Int64,1},tnumc::Array{Int64,1},jmax)
    lb=length(tnumb)
    lc=length(tnumc)
    timeb=p[3]+p[2]*(tnumb-1)#linear ephemerus
    timec=p[8]+p[7]*(tnumc-1)
    alpha0 = abs(p[2]/p[7])^(2//3)
    b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)
    # Set up planets planar-planet types for the inner and outer planets:
    planet1=TTVFaster.Planet_plane_hk(p[1],p[2],p[3],p[4],p[5])
    planet2=TTVFaster.Planet_plane_hk(p[6],p[7],p[8],p[9],p[10])

    dtb=Vector{T}(lb)
    dtc=Vector{T}(lc)
    f1=Array(T,jmax+2,5)
    f2=Array(T,jmax+2,5)
    b=Array(T,jmax+2,3)

    compute_inner_ttv!(jmax, planet1, planet2, timeb, dtb, f1,f2, b, alpha0, b0)
    compute_outer_ttv!(jmax, planet1, planet2, timec, dtc, f1,f2, b, alpha0, b0)
    return dtb,dtc
end

nb=collect(1:200)
nc=collect(1:150)

ttvb10,ttvc10=calcTTV(pinit,nb,nc,10)
ttvb7,ttvc7=calcTTV(pinit,nb,nc,7)
ttvb5,ttvc5=calcTTV(pinit,nb,nc,5)
ttvb4,ttvc4=calcTTV(pinit,nb,nc,4)
ttvb3,ttvc3=calcTTV(pinit,nb,nc,3)
ttvb2,ttvc2=calcTTV(pinit,nb,nc,2)

using Plots
plotly()

bplot=plot(nb,ttvb10,label="jmax=10")
plot!(nb,ttvb7,label="7")
plot!(nb,ttvb5,label="5")
plot!(nb,ttvb4,label="4")
plot!(nb,ttvb3,label="3")
plot!(nb,ttvb2, label="2")

cplot=plot(nc,ttvc10,label="jmax=10")
plot!(nc,ttvc7,label="7")
plot!(nc,ttvc5,label="5")
plot!(nc,ttvc4,label="4")
plot!(nc,ttvc3,label="3")
plot!(nc,ttvc2, label="2")

doubleplot=plot(bplot,cplot,layout=(2,1))

#residuals with respect to jmax=10
bres=scatter(nb, ttvb10-ttvb7,label="7", markersize=2.5)
scatter!(nb, ttvb10-ttvb5,label="5",markersize=2.5)
scatter!(nb, ttvb10-ttvb4,label="4",markersize=2.5)

cres=scatter(nc, ttvc10-ttvc7,label="7",markersize=2.5)
scatter!(nc, ttvc10-ttvc5,label="5",markersize=2.5)
scatter!(nc, ttvc10-ttvc4,label="4",markersize=2.5)
scatter!(nc, ttvc10-ttvc3,label="3",markersize=2.5)

resplot=plot(bres,cres,layout=(2,1))
