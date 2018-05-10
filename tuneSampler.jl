#automatic tuning
#include("MCMCdiagnostics.jl")
#requires TTVmodel3.jl, pstart
#samplerfunc is sampler constructor as a function of single argument
#plogtarget, pgradlogtarget, and ptensorlogtarget are functions for the target distribution
#defined in TTVmodel3.jl
#narrow tells tuner to narrow the range of tuning parameter steps to those corresponding to between 10-90% acceptance
function tuneSampler(samplerfunc,plogtarget,pgradlogtarget,ptensorlogtarget;
     start=-1.0, stop=0.7, numtune=10,nstep=5000, GAMCtuner=false, narrow=1, verbose=true)
    p= BasicContMuvParameter(:p,
      logtarget=plogtarget,
      gradlogtarget=pgradlogtarget,
      tensorlogtarget=ptensorlogtarget)

    model= likelihood_model(p, false)

    zstart=to_z(pstart)
    p0= Dict(:p=>zstart)
    mcrange= BasicMCRange(nsteps=nstep)
    smallrange=BasicMCRange(nsteps=1000)

    outopts = Dict{Symbol, Any}(:monitor=>[:value],
      :diagnostics=>[:accept])

    if !GAMCtuner
        MCtuner=VanillaMCTuner(verbose=true)
    else
        MCtuner=GAMCMCTuner(
          VanillaMCTuner(verbose=false),
          VanillaMCTuner(verbose=false),
          VanillaMCTuner(verbose=true)
        )
    end

    ndim=length(zstart)
    driftsteps= logspace(start,stop,numtune)
    aclengths=Array{Float64}(numtune, ndim)
    ess_array=Array{Float64}(numtune, ndim)
    accrate=Vector{Float64}(numtune)

    while narrow>0
        if verbose
            println("Narrow: ", narrow)
        end
        acprev=1.0
        startcalled=false #record only first time it drops below 0.9 acceptance
        for k in 1:numtune
            mcsampler=samplerfunc(driftsteps[k])
            job=BasicMCJob(model,mcsampler,smallrange, p0, tuner=MCtuner, outopts=outopts)
            run(job)
            acc=output(job).diagnosticvalues
            accrate[k]=mean(acc)
            if accrate[k]<0.9 && acprev>0.9 && (!startcalled)
                if accrate[k]>0.5
                    start=log10(driftsteps[k])
                    startcalled=true
                elseif k-1>0
                    start=log10(driftsteps[k-1])
                    startcalled=true
                    stop=log10(driftsteps[k])
                end
                if verbose
                    println("Acceptance: ", 100*accrate[k], ", start=", start)
                end
            elseif accrate[k]<0.1 && acprev>0.1
                #=if start!=log10(driftsteps[k])
                    stop=log10(driftsteps[k])
                elseif k+1<=numtune
                    stop=log10(driftsteps[k+1])
                end=#
                stop=log10(driftsteps[k])
                if verbose
                    println("Acceptance: ", 100*accrate[k], ", stop=", stop)
                end
            end
            acprev=accrate[k]
        end
        if(start==stop)
            println("Start=Stop trying narrowing again")
            continue
        end
        driftsteps=logspace(start,stop,numtune)
        narrow=narrow-1
    end

    if verbose
        println("Now tuning")
    end
    for i in 1:numtune
        mcsampler=samplerfunc(driftsteps[i])
        job=BasicMCJob(model,mcsampler,mcrange, p0, tuner=MCtuner, outopts=outopts)
        if verbose
            println("Job: ", i," , Step size: ", driftsteps[i])
        end
        run(job)
        outval=output(job).value
        acc=output(job).diagnosticvalues
        accrate[i]=mean(acc)
        if verbose
            println("Net Acceptance: ", 100.0*accrate[i],  "\%")
        end
        aclengths[i,:]=aclength(outval, threshold=0.1, maxit=nstep, jump=1,useabs=true)
        ess_array[i,:]=nstep./aclengths[i,:]
    end

    maxac=Vector(numtune)
    miness=Vector(numtune)
    for i in 1:numtune
        if(any(isnan.(aclengths[i,:])) || accrate[i]<0.1 ||accrate[i]>0.9)
            maxac[i]=Inf
            miness[i]=0.0
        else
            maxac[i]=maximum(aclengths[i,:])
            miness[i]=minimum(ess_array[i,:])
        end
    end
    minind=indmin(maxac)
    minstep=driftsteps[minind]

    maxind=indmax(miness)
    maxstep=driftsteps[maxind]

    tuneout=Dict("minstep"=>minstep,"driftsteps"=>driftsteps,
        "ess_array"=>ess_array, "aclengths"=>aclengths, "accrate"=>accrate)
    return tuneout
end

#using Plots.jl
function plotTune(tune::Dict; ndim=10)
    acplot=scatter(tune["driftsteps"],tune["aclengths"],
      layout=ndim,
      xaxis=( :log10),
      title=["1" "2" "3" "4" "5" "6" "7" "8" "9" "10"],
      leg=false)

    essplot=scatter(tune["driftsteps"],tune["ess_array"],
      layout=ndim,
      xaxis=( :log10),
      title=["1" "2" "3" "4" "5" "6" "7" "8" "9" "10"],
      leg=false)

    acceptplot= scatter(tune["driftsteps"], tune["accrate"],
      xaxis=:log10,
      xlabel="Step size",
      ylabel="Net acceptance rate",
      leg=false)

    return acplot, essplot, acceptplot
end
