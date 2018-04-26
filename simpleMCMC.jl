#testing samplers on simpler case
using Klara, Distributions
#=
means=Vector([2.0,-4.0])
sigma=Matrix{Float64}(2,2)
sigma[1,1]=1.0
sigma[2,2]=2.5
sigma[2,1]=0.25
sigma[1,2]=0.25

invsig=inv(sigma)

function plogtarget(z::Vector{Float64})
  tot= -0.5*(z-means)' * invsig * (z-means)
  #println(tot[1])
  return tot[1]
end

function pgradlogtarget(z::Vector{Float64})
  #println(-invsig * (z-means))
  return -invsig * (z-means)
end
=#

means=20.0*rand(Float64,10)-10.0
sigma= eye(10)

dist=MvNormal(means,sigma)

p= BasicContMuvParameter(
    :p, 
    logtarget= x::Vector{Float64} -> logpdf(dist, x ),
    gradlogtarget= x::Vector{Float64} -> gradlogpdf(dist, x)
)

model= likelihood_model(p, false)

sampler=MALA(0.6)
#sampler=MH(sigma)
#sampler=AM(1.0,2) #what are inputs here?
#sampler=HMC(0.1,10)
#sampler=NUTS(0.4, maxndoublings=7) #works with MV gaussian


p0= Dict(:p=>zeros(10))

mcrange= BasicMCRange(nsteps=10000,burnin=1000)

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

MCMCtuner= AcceptanceRateMCTuner(0.6, verbose=true)
#MCMCtuner=VanillaMCTuner(verbose=true)

job=BasicMCJob(model,sampler,mcrange, p0, tuner=MCMCtuner, outopts=outopts)

run(job)

out=output(job)
outdata=out.value
outdata1=outdata[:,1:3000]
outdata2=outdata[:,3001:6000]
outdata3=outdata[:,6001:9000]

using Plots
plotly()

plotfull=histogram2d(outdata[1,:],outdata[2,:],nbins=40, fc= :plasma, title="Full Dataset")
plot1=histogram2d(outdata1[1,:],outdata1[2,:],nbins=40, fc= :plasma)
plot2=histogram2d(outdata2[1,:],outdata2[2,:],nbins=40, fc= :plasma)
plot3=histogram2d(outdata3[1,:],outdata3[2,:],nbins=40, fc= :plasma)

plot(plotfull,plot1,plot2,plot3)