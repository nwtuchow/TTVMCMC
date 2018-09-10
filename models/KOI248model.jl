#julia 6 compatible
#TTVFaster statistical model for KOI248 analog
#takes B = sigma^(1/2) externally (Lower diagonal)
#takes pmeans externally

#using ForwardDiff
include("../utils/TTVfunctions.jl")

pinit=readdlm("../outputs/KOI248ptrue.txt",',')
pinit=vec(pinit)

pguess=copy(pinit)
pguess[1]=3.0e-5
pguess[4]=0.04
pguess[5]=-0.02
pguess[6]=2.0e-5
pguess[9]=0.03
pguess[10]=-0.008

bData=readdlm("../outputs/KOI248bData.txt",',')
cData=readdlm("../outputs/KOI248cData.txt",',')

np=length(pinit)

# z:transformed parameter array
zinit=to_z(pinit)
zguess=to_z(pguess)

include("TTVmodel.jl")
