#julia 6 compatible
#TTVFaster statistical model
#takes B = sigma^(1/2) externally (Lower diagonal)
#takes pmeans externally

#using ForwardDiff
include("../utils/TTVfunctions.jl")

pinit=readdlm("../outputs/KOI1270ptrue.txt",',')
pinit=vec(pinit)

pguess=copy(pinit)
pguess[1]=1.0e-6
pguess[4]= -0.01
pguess[5]= -0.01
pguess[6]=1.7e-5
pguess[9]=  -0.03
pguess[10]= -0.008

bData=readdlm("../outputs/KOI1270bData.txt",',')
cData=readdlm("../outputs/KOI1270cData.txt",',')

np=length(pinit)

# z:transformed parameter array
zinit=to_z(pinit)
zguess=to_z(pguess)

include("TTVmodel.jl")
