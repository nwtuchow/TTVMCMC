#simulate TTVFaster data set
include("TTVfunctions3.jl")
candidatedata=readdlm("../CandidateTTVs.txt",skipstart=20)
nentry=size(candidatedata)[1]

k49bData=Array{Number}(0,4)
k49cData=Array{Number}(0,4)
for i in 1:nentry
    if candidatedata[i,1]=="KOI-248.01"
        k49bData=cat(1,k49bData, candidatedata[i,2:5]')
    end
    if candidatedata[i,1]=="KOI-248.02"
        k49cData=cat(1,k49cData, candidatedata[i,2:5]')
    end
end
#header? "Number,  time(d), O-C(d), dTTV(d)"
writedlm("../KOI-248.01.txt",k49bData,",")
writedlm("../KOI-248.02.txt",k49cData,",")

P_b =7.2040
P_c =10.9123

ti_b=780.4529
ti_c=790.3470

k_b= 0.011
k_c= 0.006

h_b= 0.037
h_c= 0.027

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

mu_b=9.16 # Me/Msun
mu_c=5.91

mu_b=3.003e-6*mu_b
mu_c=3.003e-6*mu_c

pinit=[mu_b,P_b,ti_b,k_b,h_b,mu_c,P_c,ti_c,k_c,h_c]

num_b=round.(Int64,(k49bData[:,2]-pinit[3])/pinit[2] +1)
err_b=mean(k49bData[:,4])*24*60 #to minutes

num_c=round.(Int64,(k49cData[:,2]-pinit[8])/pinit[7] +1)
err_c=mean(k49bData[:,4])*24*60 #to minutes
bdata=Array{Float64}(length(num_b),3)
cdata=Array{Float64}(length(num_c),3)

simDataset!(pinit,bdata,cdata, noiseb=err_b, noisec=err_c,num_b=num_b,num_c=num_c)

writedlm("KOI248bData.txt",bdata,",")
writedlm("KOI248cData.txt",cdata,",")
writedlm("KOI248ptrue.txt",pinit,",")

using Plots
plotly()
bplot=scatter(k49bData[:,2],k49bData[:,3],yerror=k49bData[:,4], label="Data")
scatter!(bdata[:,1],bdata[:,2],yerror=bdata[:,3], label="simulated")

cplot=scatter(k49cData[:,2],k49cData[:,3],yerror=k49cData[:,4], label="Data")
scatter!(cdata[:,1],cdata[:,2],yerror=cdata[:,3], label="simulated")

doubleplot=plot(bplot,cplot, layout=(2,1), leg=true)
