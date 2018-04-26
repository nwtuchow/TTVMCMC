
#make all these plots using SMMALA tuning outputs for scaled two sin harmonic tuning

include("MCMCdiagnostics.jl")

chainSMMALA=readdlm("../../../Documents/Exoplanet_ttv_data/values_scaled2sinusoidSMMALAminstep.txt",',')

chainSMMALA[5,:]=chainSMMALA[5,:]/1000.0
chainSMMALA[10,:]=chainSMMALA[10,:]/1000.0
chainMAMALA=readdlm("sampleMAMALAchain.txt",',')
acceptSMMALA=readdlm("../outputs/accept_SMMALA_2sinusoid.txt",',')
aclengthSMMALA=readdlm("../outputs/aclength_SMMALA_2sinusoid.txt",',')
essSMMALA=readdlm("../outputs/ess_SMMALA_2sinusoid.txt",',')
stepSMMALA=readdlm("../outputs/step_SMMALA_2sinusoid.txt",',')

#=
using PyPlot

acceptplot=plot(stepSMMALA,acceptSMMALA,
    linestyle="none",
    marker="o",
    markersize=8.0,
    color="blue")
xlabel("SMMALA Driftstep")
ylabel("Net Acceptance Rate")
xscale("log")
axis([0.05,20,-0.05,1.05])
grid(true)

acplot, ax1=subplots(nrows=2,ncol=5)=#

#Alternative using Plots.jl

using Plots
plotly()

acplot=Plots.scatter(stepSMMALA,aclengthSMMALA,
  layout=10,
  xaxis=( :log10),
  title=["1" "2" "3" "4" "5" "6" "7" "8" "9" "10"],
  xlabel=["" "" "" "" "" "" "" "" "SMMALA step" ""],
  ylabel=["" "" "" "" "" "" "" "" "Autocorrelation Length" ""],
  guidefont=font(10, "Arial"),
  leg=false)

#how do I set x labels for entire plot?

essplot=Plots.scatter(stepSMMALA,essSMMALA,
  layout=10,
  xaxis=( :log10),
  title=["1" "2" "3" "4" "5" "6" "7" "8" "9" "10"],
  xlabel=["" "" "" "" "" "" "" "" "SMMALA step" ""],
  ylabel=["" "" "" "" "" "" "" "" "ESS" ""],
  guidefont=font(10, "Arial"),
  leg=false)

acceptplot= Plots.scatter(stepSMMALA, acceptSMMALA,
  xaxis=:log10,
  xlabel="Step size",
  guidefont=font(15, "Arial"),
  ylabel="Net acceptance rate",
  leg=false)

r=collect(1:200)
acfunc=autocorr(r,chainMAMALA)
thress=0.05*ones(200)

acfuncPlot=Plots.plot(r,acfunc',
    label=["1" "2" "3" "4" "5" "6" "7" "8" "9" "10"],
    xlabel="Lag",
    ylabel="Autocorrelation Function",
    linewidth=2.0)

Plots.plot!(r,thress,
    label="Threshold",
    linestyle=:dash,
    linecolor=:black,
    linewidth=3.0)

using PyPlot
using PyCall
using LaTeXStrings
@pyimport corner


cornerSMMALA=corner.corner(chainSMMALA',
    labels=["p_1", "p_2", "p_3", "p_4", "p_5", "p_6", "p_7", "p_8", "p_9", "p_{10}"],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=true)

TTVlongchain=readdlm("../../../Documents/Exoplanet_ttv_data/values_transformedTTVFasterMALAminstep3.txt",',')

outval=TTVlongchain[:,250001:end]

iter=3000000 +10*(1:100:length(outval))
#make trace plots

Plots.scatter(iter,outval[1,1:100:end],
    ylabel="mu_b",
    leg=false)

corner.corner(outval', labels=[L"\mathbf{\mu_b}",L"\mathbf{P_b}",L"\mathbf{t_{i,b}}",L"\mathbf{k_b}",L"\mathbf{h_b}",L"\mathbf{\mu_c}",L"\mathbf{P_c}",L"\mathbf{t_{i,c}}",L"\mathbf{k_c}",L"\mathbf{h_c}"],
quantiles=[0.16, 0.5, 0.84],
show_titles=true)

pmeans=zeros(10)
scale=eye(10)
include("TTVmodel.jl")
const jmax = 5
bdata=bData
cdata=cData

function fb{Tp<:Number}(p::Vector{Tp})  #gives TTVb array
    tb=bdata[:,1]
    tc=cdata[:,1]
    dtb=bdata[:,2]
    dtc=cdata[:,2]
    eb=bdata[:,3]
    ec=cdata[:,3]

    #rowe et al ephemeruses
    timeb= tb-dtb
    timec= tc-dtc
    alpha0 = abs(p[2]/p[7])^(2//3) #should these be calculated outside instead?
    b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)
    planet1=TTVFaster.Planet_plane_hk(p[1],p[2],p[3],p[4],p[5])
    planet2=TTVFaster.Planet_plane_hk(p[6],p[7],p[8],p[9],p[10])
    altttvb=Array(Tp,length(tb))
# alternative arrays as workspace
    altf1=Array(Tp,jmax+2,5)
    altf2=Array(Tp,jmax+2,5)
    altb=Array(Tp,jmax+2,3)
    #println("\naltb: ", typeof(altb))
    compute_inner_ttv!(jmax,planet1,planet2,timeb,altttvb,altf1,altf2,altb,alpha0,b0)
    return altttvb
end

function fc{Tp<:Number}(p::Vector{Tp})
    tb=bdata[:,1]
    tc=cdata[:,1]
    dtb=bdata[:,2]
    dtc=cdata[:,2]
    eb=bdata[:,3]
    ec=cdata[:,3]

    #rowe et al ephemeruses
    timeb= tb-dtb
    timec= tc-dtc
    alpha0 = abs(p[2]/p[7])^(2//3) #should these be calculated outside instead
    b0 = TTVFaster.LaplaceCoefficients.initialize(jmax+1,alpha0)
    planet1=TTVFaster.Planet_plane_hk(p[1],p[2],p[3],p[4],p[5])
    planet2=TTVFaster.Planet_plane_hk(p[6],p[7],p[8],p[9],p[10])
    altttvc=Array(Tp,length(tc))
  # alternative arrays as workspace
    altf1=Array(Tp,jmax+2,5)
    altf2=Array(Tp,jmax+2,5)
    altb=Array(Tp,jmax+2,3)
    compute_outer_ttv!(jmax,planet1,planet2,timec,altttvc,altf1,altf2,altb,alpha0,b0)
    return altttvc
end

tb=bdata[:,1]
tc=cdata[:,1]
dtb=bdata[:,2]
dtc=cdata[:,2]
eb=bdata[:,3]
ec=cdata[:,3]

#rowe et al ephemeruses
timeb= tb-dtb
timec= tc-dtc

ttvb=fb(pinit)
ttvc=fc(pinit)

pmed= [2.81e-5, 10.5, 785.62, 0.045, -0.012, 1.17e-5, 13.003, 787.69, 0.047, -0.012]
ttvbmed=fb(pmed)
ttvcmed=fc(pmed)

plotb= scatter(bData[:,1],1440*bData[:,2],
        ylabel= "TTV (min)",
        yerror=1440*bData[:,3],
        markersize=3,
        markershape=:circle,
        title="Planet b")

plot!(timeb,1440*ttvb,
        linewidth=3,
        linecolor=:red,
        label="true values")

plot!(timeb,1440*ttvbmed,
        linewidth=3,
        linecolor=:blue,
        label="recovered")


plotc= scatter(cData[:,1],1440*cData[:,2],
        xlabel= "Transit Time",
        ylabel= "TTV (min)",
        yerror=1440*cData[:,3],
        markersize=3,
        markershape=:circle,
        title="Planet c")

plot!(timec,1440*ttvc,
        linewidth=3,
        linecolor=:red,
        label="true values")

plot!(timec,1440*ttvcmed,
        linewidth=3,
        linecolor=:blue,
        label="recovered")
doubleplot=plot(plotb,plotc, layout=(2,1),leg=true, guidefont=font(10, "Arial"))

include("twoSinusoidModel.jl")

xarr=trueData[:,1]
y=twosinharmonicmodel(xarr,pinit)

SSMdataplot= Plots.scatter(trueData[:,1], trueData[:,2],
    yerror=trueData[:,3],
    ylabel="f(t,p)",
    xlabel="t",
    markersize=3,
    markershape=:circle,
    leg=false,
    guidefont=font(12, "Arial"))

Plots.plot!(xarr, y,
    linewidth=3,
    linecolor=:red)
