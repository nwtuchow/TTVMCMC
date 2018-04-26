#simulate TTVFaster data set
#based of Kepler 57 system
include("TTVfunctions3.jl")
candidatedata=readdlm("../CandidateTTVs.txt",skipstart=20)
nentry=size(candidatedata)[1]

k57bData=Array{Number}(0,4)
k57cData=Array{Number}(0,4)
for i in 1:nentry
    if candidatedata[i,1]=="KOI-1270.01"
        k57bData=cat(1,k57bData, candidatedata[i,2:5]')
    end
    if candidatedata[i,1]=="KOI-1270.02"
        k57cData=cat(1,k57cData, candidatedata[i,2:5]')
    end
end
#header? "Number,  time(d), O-C(d), dTTV(d)"
writedlm("../KOI-1270.01.txt",k57bData,",")
writedlm("../KOI-1270.02.txt",k57cData,",")

P_b =5.7295
P_c =11.6065

ti_b=781.9966
ti_c=786.7562

k_b= 0.018
k_c= 0.030

h_b= -0.019
h_c= -0.036

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

mu_b=27.81 # Me/Msun
mu_c=6.62

mu_b=3.003e-6*mu_b
mu_c=3.003e-6*mu_c

pinit=[mu_b,P_b,ti_b,k_b,h_b,mu_c,P_c,ti_c,k_c,h_c]

num_b=round(Int64,(k57bData[:,2]-pinit[3])/pinit[2] +1)
err_b=mean(k57bData[:,4])*24*60 #to minutes

num_c=round(Int64,(k57cData[:,2]-pinit[8])/pinit[7] +1)
err_c=mean(k57bData[:,4])*24*60 #to minutes
bdata=Array{Float64}(length(num_b),3)
cdata=Array{Float64}(length(num_c),3)

simDataset!(pinit,bdata,cdata, noiseb=err_b, noisec=err_c,num_b=num_b,num_c=num_c)

writedlm("KOI1270bData.txt",bdata,",")
writedlm("KOI1270cData.txt",cdata,",")
writedlm("KOI1270ptrue.txt",pinit,",")

using Plots
plotly()
bplot=scatter(k57bData[:,2],k57bData[:,3],yerror=k57bData[:,4], label="Data")
scatter!(bdata[:,1],bdata[:,2],yerror=bdata[:,3], label="simulated")

cplot=scatter(k57cData[:,2],k57cData[:,3],yerror=k57cData[:,4], label="Data")
scatter!(cdata[:,1],cdata[:,2],yerror=cdata[:,3], label="simulated")

doubleplot=plot(bplot,cplot, layout=(2,1), leg=true)
