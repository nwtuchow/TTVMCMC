#julia 6 compatible
#Noisey Kepler 307 model
#takes B = sigma^(1/2) externally (Lower diagonal)
#takes pmeans externally

#using ForwardDiff
include("../utils/TTVfunctions.jl")

pinit=readdlm("../outputs/Noisyptrue.txt",',')
pinit=vec(pinit)

pguess=copy(pinit)
pguess[1]+=2.0e-6
pguess[4]+=0.005
pguess[5]+=0.002
pguess[6]-=1.0e-6
pguess[9]-=0.004
pguess[10]+=0.001

bData=readdlm("../outputs/NoisybData.txt",',')
cData=readdlm("../outputs/NoisycData.txt",',')

np=length(pinit)

# z:transformed parameter array
zinit=to_z(pinit)
zguess=to_z(pguess)

include("TTVmodel.jl")
