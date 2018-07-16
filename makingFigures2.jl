#making figures for updated ttv paper
using Plots
plotly()
#jmax0=5
pmeans=zeros(10)
B=eye(10)
include("sinusoidFunctions2.jl")
include("TTVfunctions3old.jl")
########################
#SSM data plot
SSMptrue=readdlm("pf_bData2.txt", ',')
SSMptrue=vec(SSMptrue)
SSMbData=readdlm("sinHarmonicFit_bData2.txt", ',')
SSMcData=readdlm("sinHarmonicFit_cData2.txt", ',')


tnumb=round(Int64,SSMbData[:,1])
SSMTTVb= linsinharmonicb2(tnumb,SSMptrue)
SSMtlinb=SSMptrue[1]+(tnumb-1)*SSMptrue[2]

tnumc=round(Int64,SSMcData[:,1])
SSMTTVc=linsinharmonicc2(tnumc,SSMptrue)
SSMtlinc= SSMptrue[7]+(tnumc-1)*SSMptrue[8]

SSMplotb=scatter(SSMbData[:,2],1440*(SSMbData[:,2]-SSMtlinb),
        ylabel= "TTV (min)",
        yerror=1440*SSMbData[:,3],
        markersize=3,
        markershape=:circle,
        title="Inner Planet")

plot!(SSMTTVb,1440*(SSMTTVb-SSMtlinb),
        linewidth=3,
        linecolor=:red)

SSMplotc=scatter(SSMcData[:,2],1440*(SSMcData[:,2]-SSMtlinc),
        xlabel= "time (d)",
        ylabel= "TTV (min)",
        yerror=1440*SSMcData[:,3],
        markersize=3,
        markershape=:circle,
        title="Outer Planet")

plot!(SSMTTVc,1440*(SSMTTVc-SSMtlinc),
        linewidth=3,
        linecolor=:red)

SSMdoubleplot=plot(SSMplotb,SSMplotc, layout=(2,1),leg=false, guidefont=font(14, "Arial"))
#######################################
#kepler 307 data plot
k307ptrue=readdlm("TTVFasterptrue.txt",',')
k307ptrue=vec(k307ptrue)
k307bData=readdlm("TTVFasterbData.txt",',')
k307cData=readdlm("TTVFastercData.txt",',')

tnumb=round(Int64, (k307bData[:,1]-k307ptrue[3])/k307ptrue[2] +1)
tnumc=round(Int64, (k307cData[:,1]-k307ptrue[8])/k307ptrue[7] +1)

tlinb= k307ptrue[3]+ k307ptrue[2]*(tnumb-1)
tlinc= k307ptrue[8]+ k307ptrue[7]*(tnumc-1)

alpha0 = abs(k307ptrue[2]/k307ptrue[7])^(2//3) #should these be calculated outside instead?
b0 = TTVFaster.LaplaceCoefficients.initialize(jmax0+1,alpha0)
k307timeb=ftimeb(k307ptrue,tnumb,alpha0,b0,jmax=jmax0)
k307timec=ftimec(k307ptrue,tnumc,alpha0,b0,jmax=jmax0)

k307plotb=scatter(k307bData[:,1],1440*k307bData[:,2],
        ylabel= "TTV (min)",
        yerror=1440*k307bData[:,3],
        markersize=3,
        markershape=:circle,
        title="Inner Planet")

plot!(k307timeb,1440*(k307timeb-tlinb),
        linewidth=3,
        linecolor=:red)

k307plotc=scatter(k307cData[:,1],1440*k307cData[:,2],
        xlabel="time (d)",
        ylabel= "TTV (min)",
        yerror=1440*k307cData[:,3],
        markersize=3,
        markershape=:circle,
        title="Outer Planet")

plot!(k307timec,1440*(k307timec-tlinc),
        linewidth=3,
        linecolor=:red)

k307doubleplot=plot(k307plotb,k307plotc, layout=(2,1),leg=false, guidefont=font(14, "Arial"))
#####################################################
#Noisy kepler 307 data plot
Noisyptrue=readdlm("Noisyptrue.txt",',')
Noisyptrue=vec(Noisyptrue)
NoisybData=readdlm("NoisybData.txt",',')
NoisycData=readdlm("NoisycData.txt",',')

tnumb=round(Int64, (NoisybData[:,1]-Noisyptrue[3])/Noisyptrue[2] +1)
tnumc=round(Int64, (NoisycData[:,1]-Noisyptrue[8])/Noisyptrue[7] +1)

tlinb= Noisyptrue[3]+ Noisyptrue[2]*(tnumb-1)
tlinc= Noisyptrue[8]+ Noisyptrue[7]*(tnumc-1)

alpha0 = abs(Noisyptrue[2]/Noisyptrue[7])^(2//3)
b0 = TTVFaster.LaplaceCoefficients.initialize(jmax0+1,alpha0)
Noisytimeb=ftimeb(Noisyptrue,tnumb,alpha0,b0,jmax=jmax0)
Noisytimec=ftimec(Noisyptrue,tnumc,alpha0,b0,jmax=jmax0)

Noisyplotb=scatter(NoisybData[:,1],1440*NoisybData[:,2],
        ylabel= "TTV (min)",
        yerror=1440*NoisybData[:,3],
        markersize=3,
        markershape=:circle,
        title="Inner Planet")

plot!(Noisytimeb,1440*(Noisytimeb-tlinb),
        linewidth=3,
        linecolor=:red)

Noisyplotc=scatter(NoisycData[:,1],1440*NoisycData[:,2],
        xlabel="time (d)",
        ylabel= "TTV (min)",
        yerror=1440*NoisycData[:,3],
        markersize=3,
        markershape=:circle,
        title="Outer Planet")

plot!(Noisytimec,1440*(Noisytimec-tlinc),
        linewidth=3,
        linecolor=:red)

Noisydoubleplot=plot(Noisyplotb,Noisyplotc, layout=(2,1),leg=false, guidefont=font(14, "Arial"))
#########################################################
#kepler 49 model data plot
k49ptrue=readdlm("KOI248ptrue.txt",',')
k49ptrue=vec(k49ptrue)
k49bData=readdlm("KOI248bData.txt",',')
k49cData=readdlm("KOI248cData.txt",',')

tnumb=round(Int64, (k49bData[:,1]-k49ptrue[3])/k49ptrue[2] +1)
tnumc=round(Int64, (k49cData[:,1]-k49ptrue[8])/k49ptrue[7] +1)

tlinb= k49ptrue[3]+ k49ptrue[2]*(tnumb-1)
tlinc= k49ptrue[8]+ k49ptrue[7]*(tnumc-1)

alpha0 = abs(k49ptrue[2]/k49ptrue[7])^(2//3)
b0 = TTVFaster.LaplaceCoefficients.initialize(jmax0+1,alpha0)
k49timeb=ftimeb(k49ptrue,tnumb,alpha0,b0,jmax=jmax0)
k49timec=ftimec(k49ptrue,tnumc,alpha0,b0,jmax=jmax0)

k49plotb=scatter(k49bData[:,1],1440*k49bData[:,2],
        ylabel= "TTV (min)",
        yerror=1440*k49bData[:,3],
        markersize=3,
        markershape=:circle,
        title="Inner Planet")

plot!(k49timeb,1440*(k49timeb-tlinb),
        linewidth=3,
        linecolor=:red)

k49plotc=scatter(k49cData[:,1],1440*k49cData[:,2],
        xlabel="time (d)",
        ylabel= "TTV (min)",
        yerror=1440*k49cData[:,3],
        markersize=3,
        markershape=:circle,
        title="Outer Planet")

plot!(k49timec,1440*(k49timec-tlinc),
        linewidth=3,
        linecolor=:red)

k49doubleplot=plot(k49plotb,k49plotc, layout=(2,1),leg=false, guidefont=font(14, "Arial"))
##################################################
#kepler 57 data plot
k57ptrue=readdlm("KOI1270ptrue.txt",',')
k57ptrue=vec(k57ptrue)
k57bData=readdlm("KOI1270bData.txt",',')
k57cData=readdlm("KOI1270cData.txt",',')

tnumb=round(Int64, (k57bData[:,1]-k57ptrue[3])/k57ptrue[2] +1)
tnumc=round(Int64, (k57cData[:,1]-k57ptrue[8])/k57ptrue[7] +1)

tlinb= k57ptrue[3]+ k57ptrue[2]*(tnumb-1)
tlinc= k57ptrue[8]+ k57ptrue[7]*(tnumc-1)

alpha0 = abs(k57ptrue[2]/k57ptrue[7])^(2//3)
b0 = TTVFaster.LaplaceCoefficients.initialize(jmax0+1,alpha0)
k57timeb=ftimeb(k57ptrue,tnumb,alpha0,b0,jmax=jmax0)
k57timec=ftimec(k57ptrue,tnumc,alpha0,b0,jmax=jmax0)

k57plotb=scatter(k57bData[:,1],1440*k57bData[:,2],
        ylabel= "TTV (min)",
        yerror=1440*k57bData[:,3],
        markersize=3,
        markershape=:circle,
        title="Inner Planet")

plot!(k57timeb,1440*(k57timeb-tlinb),
        linewidth=3,
        linecolor=:red)

k57plotc=scatter(k57cData[:,1],1440*k57cData[:,2],
        xlabel="time (d)",
        ylabel= "TTV (min)",
        yerror=1440*k57cData[:,3],
        markersize=3,
        markershape=:circle,
        title="Outer Planet")

plot!(k57timec,1440*(k57timec-tlinc),
        linewidth=3,
        linecolor=:red)

k57doubleplot=plot(k57plotb,k57plotc, layout=(2,1),leg=false, guidefont=font(14, "Arial"))

function cornerUncertainty{T<:Number}(outval::Array{T,2}, quantiles=[0.16, 0.5, 0.84])
    numparam=size(outval)[1]
    outarr=Array{T}(numparam,3)
    for q in 1:numparam
        qls=quantile(outval[q,:],quantiles)
        outarr[q,1]=qls[2]
        outarr[q,2]=qls[3]-qls[2]
        outarr[q,3]=qls[2]-qls[1]
    end
    #row 1: median 2:upper 3:lower
    return outarr
end

######################################################################################
using PyPlot
using PyCall
using LaTeXStrings
@pyimport corner

SSMchain=readdlm("../Exoplanet_ttv_data/values_correctedSSM_HMC.txt",',')
SSMptrue=readdlm("pf_bData2.txt", ',')
SSMptrue=vec(SSMptrue)
SSMlabels=[L"\mathbf{t_{i,b}}",L"\mathbf{P_b}",L"\mathbf{A_b}",L"\mathbf{B_b}",L"\mathbf{C_b}",
    L"\mathbf{D_b}",L"\mathbf{t_{i,c}}",L"\mathbf{P_c}",L"\mathbf{A_c}",L"\mathbf{B_c}",
    L"\mathbf{C_c}",L"\mathbf{D_c}"]
cornerSSM=corner.corner(SSMchain',
    labels=SSMlabels,
    quantiles=[0.16, 0.5, 0.84],
    truths=SSMptrue,
    use_math_text=false,
    top_ticks=false,
    show_titles=false)

TTVlabels=[L"$\mu_1$ $(M_{\oplus}/M_{\odot})$",L"$P_1$",L"$t_{i,1}$",L"$k_1$",L"$h_1$",L"$\mu_2$ $(M_{\oplus}/M_{\odot})$",L"$P_2$",L"$t_{i,2}$",L"$k_2$",L"$h_2$"]

indices=[1,4,5,6,9,10]

k307chain=readdlm("../Exoplanet_ttv_data/values_transformedTTVFasterHMC.txt",',')
k307ptrue=readdlm("TTVFasterptrue.txt",',')
k307ptrue=vec(k307ptrue)

#convert to M_⊕/M_⊙
k307chain[1,:]=k307chain[1,:]/3.003e-6
k307chain[6,:]=k307chain[6,:]/3.003e-6
k307ptrue[1]=k307ptrue[1]/3.003e-6
k307ptrue[6]=k307ptrue[6]/3.003e-6
cornerk307=corner.corner(k307chain[indices,:]',
    labels=TTVlabels[indices],
    label_kwargs=Dict("fontsize"=>22.0),
    quantiles=[0.16, 0.5, 0.84],
    truths=k307ptrue[indices],
    use_math_text=true,
    top_ticks=false,
    show_titles=false,
    title_fmt=".6f")

Noisychain=readdlm("../Exoplanet_ttv_data/values_NoisyKep307MAMALA.txt",',')
Noisyptrue=readdlm("Noisyptrue.txt",',')
Noisyptrue=vec(Noisyptrue)
#convert to M_⊕/M_⊙
Noisychain[1,:]=Noisychain[1,:]/3.003e-6
Noisychain[6,:]=Noisychain[6,:]/3.003e-6
Noisyptrue[1]=Noisyptrue[1]/3.003e-6
Noisyptrue[6]=Noisyptrue[6]/3.003e-6
cornerNoisy=corner.corner(Noisychain[indices,:]',
    labels=TTVlabels[indices],
    label_kwargs=Dict("fontsize"=>22.0),
    quantiles=[0.16, 0.5, 0.84],
    truths=Noisyptrue[indices],
    use_math_text=true,
    top_ticks=false,
    show_titles=false,
    title_fmt=".6f")


k49chain=readdlm("../Exoplanet_ttv_data/values_KOI248MAMALA.txt",',')
k49ptrue=readdlm("KOI248ptrue.txt",',')
k49ptrue=vec(k49ptrue)

#convert to M_⊕/M_⊙
k49chain[1,:]=k49chain[1,:]/3.003e-6
k49chain[6,:]=k49chain[6,:]/3.003e-6
k49ptrue[1]=k49ptrue[1]/3.003e-6
k49ptrue[6]=k49ptrue[6]/3.003e-6
cornerk49=corner.corner(k49chain[indices,:]',
    labels=TTVlabels[indices],
    label_kwargs=Dict("fontsize"=>22.0),
    quantiles=[0.16, 0.5, 0.84],
    truths=k49ptrue[indices],
    use_math_text=true,
    top_ticks=false,
    show_titles=false,
    title_fmt=".6f")

k57chain=readdlm("../Exoplanet_ttv_data/values_KOI1270DEMCMC.txt",',')
k57ptrue=readdlm("KOI1270ptrue.txt",',')
k57ptrue=vec(k57ptrue)
#convert to M_⊕/M_⊙
k57chain[1,:]=k57chain[1,:]/3.003e-6
k57chain[6,:]=k57chain[6,:]/3.003e-6
k57ptrue[1]=k57ptrue[1]/3.003e-6
k57ptrue[6]=k57ptrue[6]/3.003e-6

ngen=size(k57chain)[2]
smallk57chain=k57chain[:,1:10:ngen]

cornerk57=corner.corner(smallk57chain[indices,:]',
    labels=TTVlabels[indices],
    label_kwargs=Dict("fontsize"=>22.0),
    quantiles=[0.16, 0.5, 0.84],
    truths=k57ptrue[indices],
    use_math_text=true,
    top_ticks=false,
    show_titles=false,
    title_fmt=".6f")

#####################################################
using Klara
using GAMCSampler
using Plots
plotly()

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
fmala(x)=MALA(x)

start=-3.0
stop=0.7
tune_arr=tuneSampler(fmala,plogtarget,pgradlogtarget,ptensorlogtarget,
    numtune=20,start=start,stop=stop,GAMCtuner=false,narrow=1)

writedlm("MALA_accept.txt",tune_arr["accrate"],",")
writedlm("MALA_drift.txt",tune_arr["driftsteps"],",")

acceptplot= scatter(tune_arr["driftsteps"], tune_arr["accrate"],
    xaxis=:log10,
    xlabel="Step size",
    xtickfont=font(11, "Arial"),
    ylabel="Net acceptance rate",
    ytickfont=font(11, "Arial"),
    leg=false,
    guidefont=font(14, "Arial"))
#tuneplots=plotTune(tune_arr)




######################################################################
using Klara
#using MAMALASampler
using GAMCSampler
using Plots
plotly()

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

selectsamp=[3,6,8,11]

minsteps=[0.918,0.822,0.717,0.736,1.03,0.974,0.464,0.880,0.880,0.880,0.880]

p = BasicContMuvParameter(:p,
  logtarget=plogtarget,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget)

model = likelihood_model(p, false)

zstart=to_z(pstart)
p0= Dict(:p=>zstart)

nstep=10000
mcrange= BasicMCRange(nsteps=nstep)

outopts= Dict{Symbol, Any}(:monitor=>[:value], :diagnostics=>[:accept])
outchains=Array{Float64}(length(selectsamp),10,nstep)

thress=0.05*ones(200)
r=collect(1:200)
acfuncs=Array{Float64}(length(selectsamp),10,200)

count=1
for i in selectsamp
    mcsampler=sampfuncs[i](minsteps[i])
    if useGAMC[i]
        MCtuner=GAMCTuner(
          VanillaMCTuner(verbose=false),
          VanillaMCTuner(verbose=false),
          VanillaMCTuner(verbose=true)
        )
    else
        MCtuner=VanillaMCTuner(verbose=true)
    end
    job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
    run(job)
    outval=output(job).value
    acfuncs[count,:,:]=autocorr(r,outval)
    outchains[count,:,:]=outval
    count+=1
end



mu_b_acfuncs=abs.(acfuncs[:,1,:])
writedlm("mu_b_acfuncs.txt",mu_b_acfuncs,",")
lstyles=[:solid,:dash,:dot,:dashdot]
lcolor=[:black,:red,:blue,:green]
lnames=["HMC(n=3)" "MALA" "GAMC(k=0)" "GAMC(k=1e6)"]

acfuncPlot=plot(r[1:100],thress[1:100],
    label="Threshold",
    linestyle=:dash,
    linecolor=:black,
    linewidth=3.0,
    xlabel="Lag",
    xtickfont=font(11, "Arial"),
    ylabel="Autocorrelation Function",
    ytickfont=font(11, "Arial"),
    guidefont=font(14, "Arial"),
    legendfont=font(12,"Arial"))

for j in 1:4
    plot!(r[1:100],mu_b_acfuncs[j,1:100],
        label=lnames[j],
        linewidth=1.5,
        linestyle=lstyles[j],
        linecolor=lcolor[j])
end

acls=Array{Float64}(4)
for k in 1:4
    acls[k]=aclength(outchains[k,1,:],threshold=0.05,useabs=true)
end
