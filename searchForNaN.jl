#find which regions in given parameter space return NaN in ttvfaster model

include("TTVfunctions.jl")
include("modifiedTTVFasterFunctions.jl")

#says whether TTVFaster model works for specified inputs
function TTVmodelWorks{T<:Number}(p::Vector{T};num_b::Int64=150,num_c::Int64=100)
    if !isvalidTTV(p)
        return 0
    end
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

    if any(isnan,dtb) || any(isnan, dtc)
        return 0
    else
        return 1
    end
end

#kepler 307b params
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

numvalues=5000
#kbvalues=0.002*rand(numvalues)-0.0410
kbvalues=2.0*rand(numvalues)-1.0
hbvalues=2.0*rand(numvalues)-1.0
#kcvalues=0.0015*rand(numvalues)-0.030
#kcvalues=2.0*rand(numvalues)-1.0
#Pb_values=10.4990 +0.002*rand(numvalues)
#Pc_values=12.999 + 0.002*rand(numvalues)
#Pb_values=3.0+30.0*rand(numvalues)
#Pc_values=5.0+30.0*rand(numvalues)

valid_array=Vector{Int64}(numvalues)

for i in 1:numvalues
    p = [mu_b,P_b,ti_b,kbvalues[i],hbvalues[i],mu_c,P_c,ti_c,k_c,h_c]
    #println(p)
    valid_array[i]=TTVmodelWorks(p)
end

using Plots
plotly()

scatter(kbvalues, hbvalues,
        zcolor=valid_array,
        xlabel="k_b",
        ylabel="h_c")
