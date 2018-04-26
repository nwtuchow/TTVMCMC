#fit simple sinusoid 12 param to TTVFaster data
using Optim
using ForwardDiff

include("sinusoidFunctions.jl")

verbose=false
bData=readdlm("TTVFasterbData.txt",',')
cData=readdlm("TTVFastercData.txt",',')

ndim=8
B=eye(ndim)
P_b =10.5
P_c =13.0

P_TTV=1/(5/P_c-4/P_b)


pinit=[bData[1,1],P_b,0.003,5.7,cData[1,1],P_c, 0.002,pi/2]
#yb= p[1] + (numb-1)p[2] + p[3] cos(2pi x /PTTV +p[4])
#yc= p[5] + (numb-1) p[6] + p[7] cos(2pi x /PTTV + p[8])

config= ForwardDiff.GradientConfig(pinit)

function f{T<:Number}(z::Vector{T})
  param=B*z
  tb=bData[:,1]
  dtb=bData[:,2]

  tlinb=tb-dtb
  tnumb=round(Int64,(tlinb-bData[1,1])/P_b +1)
  eb=bData[:,3]

  yb=linsinb4(tnumb,param)

  tc=cData[:,1]
  dtc=cData[:,2]

  tlinc=tc-dtc
  tnumc=round(Int64,(tlinc-cData[1,1])/P_c +1)
  ec=cData[:,3]

  yc=linsinc4(tnumc,param)

  chisqb=0.0
  for j in 1:length(tb)
    chisqb+= (yb[j]-tb[j])^2/eb[j]^2
  end

  chisqc=0.0
  for k in 1:length(tc)
    chisqc+= (yc[k]-tc[k])^2/ec[k]^2
  end

  return chisqb+chisqc
end

function g!{T<:Number}(z::Vector{T},gstore::Vector{T})
 # config= ForwardDiff.GradientConfig(x)
  ForwardDiff.gradient!(gstore,f,z,config)
  if verbose
    println("# z = ", z)
    println("# f = ", f(z))
    println("# grad(f) = ", gstore)
    flush(STDOUT)
  end
end

result = optimize(f, g!, pinit,LBFGS(), Optim.Options(show_trace=verbose, show_every=1) )
pf=result.minimizer
#resultb =optimize(fb, gb!, pinit,LBFGS(), Optim.Options(show_trace=verbose, show_every=1) )
#resultc =optimize(fc, gc!, pinit,LBFGS(), Optim.Options(show_trace=verbose, show_every=1) )
#pfb=resultb.minimizer
#pfc=resultc.minimizer

tb=bData[:,1]
tc=cData[:,1]
dtb=bData[:,2]
dtc=cData[:,2]

tlinb= tb-dtb
tlinc= tc-dtc
tnumb=round(Int64,(tlinb-bData[1,1])/P_b +1)
tnumc=round(Int64,(tlinc-cData[1,1])/P_c +1)

yfb=linsinb4(tnumb,pf)
yfc=linsinc4(tnumc,pf)

using Plots
plotly()

plot1= scatter(tb,dtb,yerror=bData[:,3])
plot!(tlinb,yfb-tlinb,linewidth=3,linecolor=:red)

plot2= scatter(tc,dtc, yerror=cData[:,3])
plot!(tlinc,yfc-tlinc,linewidth=3,linecolor=:red)

doubleplot= plot(plot1,plot2,layout=(2,1))

sinHarFitb=simDatab4(tnumb,pf, 5.0/1440.0)
sinHarFitc=simDatac4(tnumc,pf, 5.0/1440.0)

writedlm("sinHarmonicFit_bData4.txt",sinHarFitb,",")
writedlm("sinHarmonicFit_cData4.txt",sinHarFitc,",")
writedlm("pf_bData4.txt",pf,",") #actually for b and c data
