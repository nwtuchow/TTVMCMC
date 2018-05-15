#making figures for updated ttv paper
using Plots
plotly()
jmax0=5
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
        title="Planet b")

plot!(SSMTTVb,1440*(SSMTTVb-SSMtlinb),
        linewidth=3,
        linecolor=:red)

SSMplotc=scatter(SSMcData[:,2],1440*(SSMcData[:,2]-SSMtlinc),
        xlabel= "time (d)",
        ylabel= "TTV (min)",
        yerror=1440*SSMcData[:,3],
        markersize=3,
        markershape=:circle,
        title="Planet c")

plot!(SSMTTVc,1440*(SSMTTVc-SSMtlinc),
        linewidth=3,
        linecolor=:red)

SSMdoubleplot=plot(SSMplotb,SSMplotc, layout=(2,1),leg=false, guidefont=font(10, "Arial"))
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
        title="Planet b")

plot!(k307timeb,1440*(k307timeb-tlinb),
        linewidth=3,
        linecolor=:red)

k307plotc=scatter(k307cData[:,1],1440*k307cData[:,2],
        xlabel="time (d)",
        ylabel= "TTV (min)",
        yerror=1440*k307cData[:,3],
        markersize=3,
        markershape=:circle,
        title="Planet c")

plot!(k307timec,1440*(k307timec-tlinc),
        linewidth=3,
        linecolor=:red)

k307doubleplot=plot(k307plotb,k307plotc, layout=(2,1),leg=false, guidefont=font(10, "Arial"))
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
        title="Planet b")

plot!(Noisytimeb,1440*(Noisytimeb-tlinb),
        linewidth=3,
        linecolor=:red)

Noisyplotc=scatter(NoisycData[:,1],1440*NoisycData[:,2],
        xlabel="time (d)",
        ylabel= "TTV (min)",
        yerror=1440*NoisycData[:,3],
        markersize=3,
        markershape=:circle,
        title="Planet c")

plot!(Noisytimec,1440*(Noisytimec-tlinc),
        linewidth=3,
        linecolor=:red)

Noisydoubleplot=plot(Noisyplotb,Noisyplotc, layout=(2,1),leg=false, guidefont=font(10, "Arial"))
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
        title="Planet b")

plot!(k49timeb,1440*(k49timeb-tlinb),
        linewidth=3,
        linecolor=:red)

k49plotc=scatter(k49cData[:,1],1440*k49cData[:,2],
        xlabel="time (d)",
        ylabel= "TTV (min)",
        yerror=1440*k49cData[:,3],
        markersize=3,
        markershape=:circle,
        title="Planet c")

plot!(k49timec,1440*(k49timec-tlinc),
        linewidth=3,
        linecolor=:red)

k49doubleplot=plot(k49plotb,k49plotc, layout=(2,1),leg=false, guidefont=font(10, "Arial"))
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
        title="Planet b")

plot!(k57timeb,1440*(k57timeb-tlinb),
        linewidth=3,
        linecolor=:red)

k57plotc=scatter(k57cData[:,1],1440*k57cData[:,2],
        xlabel="time (d)",
        ylabel= "TTV (min)",
        yerror=1440*k57cData[:,3],
        markersize=3,
        markershape=:circle,
        title="Planet c")

plot!(k57timec,1440*(k57timec-tlinc),
        linewidth=3,
        linecolor=:red)

k57doubleplot=plot(k57plotb,k57plotc, layout=(2,1),leg=false, guidefont=font(10, "Arial"))

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

TTVlabels=[L"\mathbf{\mu_b}",L"\mathbf{P_b}",L"\mathbf{t_{i,b}}",L"\mathbf{k_b}",L"\mathbf{h_b}",L"\mathbf{\mu_c}",L"\mathbf{P_c}",L"\mathbf{t_{i,c}}",L"\mathbf{k_c}",L"\mathbf{h_c}"]
indices=[1,4,5,6,9,10]

k307chain=readdlm("../Exoplanet_ttv_data/values_transformedTTVFasterHMC.txt",',')
k307ptrue=readdlm("TTVFasterptrue.txt",',')
k307ptrue=vec(k307ptrue)
cornerk307=corner.corner(k307chain[indices,:]',
    labels=TTVlabels[indices],
    quantiles=[0.16, 0.5, 0.84],
    truths=k307ptrue[indices],
    use_math_text=true,
    top_ticks=false,
    show_titles=false,
    title_fmt=".6f")

Noisychain=readdlm("../Exoplanet_ttv_data/values_NoisyKep307MAMALA.txt",',')
Noisyptrue=readdlm("Noisyptrue.txt",',')
Noisyptrue=vec(Noisyptrue)
cornerNoisy=corner.corner(Noisychain[indices,:]',
    labels=TTVlabels[indices],
    quantiles=[0.16, 0.5, 0.84],
    truths=Noisyptrue[indices],
    use_math_text=true,
    top_ticks=false,
    show_titles=false,
    title_fmt=".6f")


k49chain=readdlm("../Exoplanet_ttv_data/values_KOI248MAMALA.txt",',')
k49ptrue=readdlm("KOI248ptrue.txt",',')
k49ptrue=vec(k49ptrue)
cornerk49=corner.corner(k49chain[indices,:]',
    labels=TTVlabels[indices],
    quantiles=[0.16, 0.5, 0.84],
    truths=k49ptrue[indices],
    use_math_text=true,
    top_ticks=false,
    show_titles=false,
    title_fmt=".6f")

k57chain=readdlm("../Exoplanet_ttv_data/values_KOI1270DEMCMC.txt",',')
k57ptrue=readdlm("KOI1270ptrue.txt",',')
k57ptrue=vec(k57ptrue)

ngen=size(k57chain)[2]
smallk57chain=k57chain[:,1:10:ngen]

cornerk57=corner.corner(smallk57chain[indices,:]',
    labels=TTVlabels[indices],
    quantiles=[0.16, 0.5, 0.84],
    truths=k57ptrue[indices],
    use_math_text=true,
    top_ticks=false,
    show_titles=false,
    title_fmt=".6f")
