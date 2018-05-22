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
tune_arr=Vector(nsamp)
minsteps=Vector(nsamp)

start=-2.0
stop=0.7

for q in 1:nsamp
    println("Sampler: ", sampnames[q])
    tune_arr[q]=tuneSampler(sampfuncs[q],plogtarget,pgradlogtarget,ptensorlogtarget,
        numtune=20,start=start,stop=stop,GAMCtuner=useGAMC[q],narrow=0)
    minsteps[q]=tune_arr[q]["minstep"]
    println("Minstep: ", minsteps[q])
end

bigessarr=tune_arr[1]["ess_array"]
for i in 2:nsamp
    bigessarr=cat(3,bigessarr,tune_arr[i]["ess_array"])
end

for j in 1:20
    for k in 1:10
        for l in 1:nsamp
            if isnan(bigessarr[j,k,l])
                bigessarr[j,k,l]=0.0
            end
        end
    end
end
driftsteps= logspace(start,stop,20)

writedlm("ess_k307.txt",bigessarr,",")
writedlm("driftsteps_k307.txt",driftsteps,",")

using Plots
plotly()

mubess=bigessarr[:,1,:]

mub_plot=plot(driftsteps,mubess,
    xaxis=("Driftstep",:log10),
    ylabel="μ_b ESS",
    label=["HMC(n=1)" "HMC(n=2)" "HMC(n=3)" "HMC(n=5)" "HMC(n=7)" "MALA" "SMMALA" "GAMC(i=0)" "GAMC(i=25000)" "GAMC(i=50000)" "GAMC(i=1e6)"],
    lw=2)
