#julia 5 version
#simulate TTVFaster data set
include("TTVfunctions3old.jl")

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

bData=readdlm("TTVFasterbData.txt",',')
cData=readdlm("TTVFastercData.txt",',')

num_b=round(Int64,(bData[:,1]-pinit[3])/pinit[2] +1)
err_b=15.0 #minutes

num_c=round(Int64,(cData[:,1]-pinit[8])/pinit[7] +1)
err_c=15.0 #minutes
bdata2=Array{Float64}(length(num_b),3)
cdata2=Array{Float64}(length(num_c),3)

simDataset!(pinit,bdata2,cdata2, noiseb=err_b, noisec=err_c,num_b=num_b,num_c=num_c)

writedlm("NoisybData.txt",bdata2,",")
writedlm("NoisycData.txt",cdata2,",")
writedlm("Noisyptrue.txt",pinit,",")

using Plots
plotly()
bplot=scatter(bData[:,1],bData[:,2],yerror=bData[:,3], label="Data")
scatter!(bdata2[:,1],bdata2[:,2],yerror=bdata2[:,3], label="simulated")

cplot=scatter(cData[:,1],cData[:,2],yerror=cData[:,3], label="Data")
scatter!(cdata2[:,1],cdata2[:,2],yerror=cdata2[:,3], label="simulated")

doubleplot=plot(bplot,cplot, layout=(2,1), leg=true)
