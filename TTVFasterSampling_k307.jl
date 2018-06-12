using Klara
#using MAMALASampler
using GAMCSampler

covTTV=readdlm("pilotCov3.txt",',')
pmeans=readdlm("pilotMeans3.txt",',')
pmeans=vec(pmeans)

pstart=readdlm("pilotLast3.txt",',')
pstart=vec(pstart)


covTTVhalf= ctranspose(chol(covTTV))
B=covTTVhalf #sigma^(1/2)

include("TTVmodel3.jl")
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
sampnames=["HMC1", "HMC2","HMC3","HMC5","HMC7","MALA","SMMALA","GAMC(i=0)","GAMC(i=25000)","GAMC(i=50000)","GAMC(i=1e6)"]
sampfuncs=[fhmc1,fhmc2,fhmc3,fhmc5,fhmc7,fmala,fsmmala,fgamc1,fgamc2,fgamc3,fgamc4]
useGAMC=[false,false,false,false,false,false,false,true,true,true,true]
tune_arr=Vector(4)
minsteps=Vector(4)

start=log10(0.05)
stop=log10(4.0)
nselect=4
selectsamp=[3,6,8,11]
count=1
for q in selectsamp
    println("Sampler: ", sampnames[q])
    tune_arr[count]=tuneSampler(sampfuncs[q],plogtarget,pgradlogtarget,ptensorlogtarget,
        numtune=30,start=start,stop=stop,GAMCtuner=useGAMC[q],narrow=0)
    minsteps[count]=tune_arr[count]["minstep"]
    println("Minstep: ", minsteps[count])
    count+=1
end

bigessarr=tune_arr[1]["ess_array"]
for i in 2:nselect
    bigessarr=cat(3,bigessarr,tune_arr[i]["ess_array"])
end

for j in 1:30
    for k in 1:10
        for l in 1:nselect
            if isnan(bigessarr[j,k,l])
                bigessarr[j,k,l]=0.0
            end
        end
    end
end
driftsteps= logspace(start,stop,30)

writedlm("ess_k307.txt",bigessarr,",")
writedlm("driftsteps_k307.txt",driftsteps,",")

bigessarr=readdlm("ess_k307.txt",',')
bigessarr=reshape(bigessarr,(30,10,4))
driftsteps=readdlm("driftsteps_k307.txt",',')


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
    "size"=>18)
ax[:set_xlabel]("Driftstep",fontdict=font1)
ax[:set_xlim]([driftsteps[1],driftsteps[end]])
ax[:set_ylabel](L"$\mu_b$ ESS", fontdict=font1)
p1=PyPlot.plot(driftsteps,mubess[:,1],
    linestyle="dashed",
    linewidth=2,
    marker="o",
    color="black",
    label="HMC(n=3)")
p2=PyPlot.plot(driftsteps,mubess[:,2],
    linestyle="-.",
    linewidth=2,
    marker="+",
    color="red",
    label="MALA")
p3=PyPlot.plot(driftsteps,mubess[:,3],
    linestyle="solid",
    linewidth=2,
    marker="x",
    color="blue",
    label="GAMC(i=0)")
p3=PyPlot.plot(driftsteps,mubess[:,4],
    linestyle="dotted",
    linewidth=2,
    marker="s",
    color="green",
    label="GAMC(i=1e6)")

legend(loc="upper left")

#alternate plot
using Plots
plotly()
lstyles=[:solid,:dash,:dot,:dashdot]
lcolor=[:black,:red,:blue,:green]
lnum= [1,4,6,9]
lnames=["μ_b" "k_b" "μ_c" "k_b"]

mala_ess=bigessarr[:,:,2]

essPlot= plot(driftsteps, mala_ess[:,lnum[1]],
    label=lnames[1],
    linestyle=lstyles[1],
    linecolor=lcolor[1],
    linewidth=1.5,
    xlabel="Driftstep",
    xaxis=:log10,
    xtickfont=font(11, "Arial"),
    ylabel="ESS",
    ytickfont=font(11, "Arial"),
    guidefont=font(14, "Arial"),
    legendfont=font(12,"Arial"))

for j in 2:4
    plot!(driftsteps, mala_ess[:,lnum[j]],
        label=lnames[j],
        linewidth=1.5,
        linestyle=lstyles[j],
        linecolor=lcolor[j])
end
