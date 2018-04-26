#example calling hessian
using ForwardDiff

include("TTVfunctions.jl")
include("../TTVFaster-noah/Julia/test_ttv.jl")
#use sample data set
bData=readdlm("../koi0248.01.tt")
cData=readdlm("../koi0248.02.tt")

#try for Kepler 49, KOI 248
#using Jontof hutter 2016 estimates for params
P_b =7.2040
P_c =10.9123

t0_b=780.4529
t0_c=790.3470

k_b= 0.011
k_c= 0.006

h_b= 0.037
h_c= 0.027 #real

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

mu_b=9.65 # Me/Msun
mu_c=6.28

mu_b=3.003e-6*mu_b
mu_c=3.003e-6*mu_c

ti_b=bData[1,1]
ti_c=cData[1,1]

alpha0 = (P_b/P_c)^(2//3)
b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)

p=[mu_b,P_b,ti_b,k_b,h_b,mu_c,P_c,ti_c,k_c,h_c]

np=length(p)
jconfig= ForwardDiff.JacobianConfig(p)
gstore=Vector{eltype(p)}(np) #store gradient
gstore2=Vector{eltype(p)}(np)
hstore=Array{eltype(p)}(np,np) #store hessian

f(x)=fittransit(bData,cData,x)

g!(x,gstore)=gradtest!(p,gstore,bData,cData,jconfig)
gconfig=ForwardDiff.GradientConfig(p)
g2!(x,gstore)=ForwardDiff.gradient!(gstore,f,x,gconfig)

#=h!(x,hstore)=hesstest!(p,hstore,bData,cData,jconfig)
h!(p,hstore)
hstore2=hessian_finite_diff(f,p,delta=1e-6,scale=[p[1],p[2],1.0,1.0,1.0,p[6],p[7],1.0,1.0,1.0])

ratio=hstore./hstore2 #comparing hessian outputs=#
#some NaNs because some terms divide by zero
