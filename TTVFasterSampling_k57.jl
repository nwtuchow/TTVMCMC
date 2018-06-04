using Klara
#using MAMALASampler
using GAMCSampler

covTTV=readdlm("KOI1270Cov.txt",',')
pmeans=readdlm("KOI1270Means.txt",',')
pmeans=vec(pmeans)

pstart=readdlm("KOI1270Last.txt",',')
pstart=vec(pstart)


covTTVhalf= ctranspose(chol(covTTV))
B=covTTVhalf #sigma^(1/2)

include("KOI1270model.jl")
include("MCMCdiagnostics.jl")

include("tuneSampler.jl")

fhmc1(x)=HMC(x,1)
fhmc2(x)=HMC(x,2)
fhmc3(x)=HMC(x,3)
fhmc5(x)=HMC(x,5)
fhmc7(x)=HMC(x,7)
fmala(x)=MALA(x)
fsmmala(x)=SMMALA(x, H -> simple_posdef(H, a=1500.))
fgamc1(x)=GAMC(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=x,
  minorscale=0.01,
  c=0.01
)
fgamc2(x)=GAMC(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+25000, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=x,
  minorscale=0.01,
  c=0.01
)
fgamc3(x)=GAMC(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+50000, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=x,
  minorscale=0.01,
  c=0.01
)
fgamc4(x)=GAMC(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i+1000000, 50000, 10.),
  transform=H -> simple_posdef(H, a=1500.),
  driftstep=x,
  minorscale=0.01,
  c=0.01
)

nsamp=11
sampnames=["HMC1", "HMC2","HMC3","HMC5","HMC7","MALA","SMMALA","GAMC(k=0)","GAMC(k=25000)","GAMC(k=50000)","GAMC(k=1e6)"]
sampfuncs=[fhmc1,fhmc2,fhmc3,fhmc5,fhmc7,fmala,fsmmala,fgamc1,fgamc2,fgamc3,fgamc4]
useGAMC=[false,false,false,false,false,false,false,true,true,true,true]
tune_arr=Vector(4)
minsteps=Vector(4)

start= -4
stop= 0.7

nselect=4
selectsamp=[3,6,8,11]
count=1
for q in selectsamp
    println("Sampler: ", sampnames[q])
    tune_arr[count]=tuneSampler(sampfuncs[q],plogtarget,pgradlogtarget,ptensorlogtarget,
        numtune=15,start=start,stop=stop,GAMCtuner=useGAMC[q],narrow=1,acc_upper=0.95,acc_lower=0.05)
    minsteps[count]=tune_arr[count]["minstep"]
    println("Minstep: ", minsteps[count])
    count+=1
end

bigessarr=tune_arr[1]["ess_array"]
driftsteps=tune_arr[1]["driftsteps"]
for i in 2:nselect
    bigessarr=cat(3,bigessarr,tune_arr[i]["ess_array"])
    driftsteps=cat(2,driftsteps,tune_arr[i]["driftsteps"])
end

for j in 1:15
    for k in 1:10
        for l in 1:nselect
            if isnan(bigessarr[j,k,l])
                bigessarr[j,k,l]=0.0
            end
        end
    end
end


writedlm("ess_k57.txt",bigessarr,",")
writedlm("driftsteps_k57.txt",driftsteps,",")

bigessarr=readdlm("ess_k57.txt",',')
bigessarr=reshape(bigessarr,(15,10,4))
driftsteps=readdlm("driftsteps_k57.txt",',')


mubess=bigessarr[:,1,:]

#mub_plot=plot(driftsteps,mubess,
#    xaxis=("Driftstep",:log10),
#    ylabel=L"$\mu_b$ ESS",
#    label=["HMC(n=3)" "MALA" "GAMC(i=0)" "GAMC(i=1e6)"],
#    ls=[:dash,:dot,:solid,:dotdash],
#    lw=2)

using PyPlot
using LaTeXStrings

fig= PyPlot.figure()
ax=axes()
ax[:set_xscale]("log")
font1=Dict("family"=>"arial",
    "color"=>"black",
    "weight"=>"normal",
    "size"=>16)
ax[:set_xlabel]("Driftstep",fontdict=font1)
ax[:set_xlim]([driftsteps[1],driftsteps[end]])
ax[:set_ylabel](L"$\mu_b$ ESS", fontdict=font1)
p1=PyPlot.plot(driftsteps[:,1],mubess[:,1],
    linestyle="dashed",
    linewidth=2,
    marker="o",
    color="black",
    label="HMC(n=3)")
p2=PyPlot.plot(driftsteps[:,2],mubess[:,2],
    linestyle="-.",
    linewidth=2,
    marker="+",
    color="red",
    label="MALA")
p3=PyPlot.plot(driftsteps[:,3],mubess[:,3],
    linestyle="solid",
    linewidth=2,
    marker="x",
    color="blue",
    label="GAMC(k=0)")
p3=PyPlot.plot(driftsteps[:,4],mubess[:,4],
    linestyle="dotted",
    linewidth=2,
    marker="s",
    color="green",
    label="GAMC(k=1e6)")

legend(loc="upper left")
