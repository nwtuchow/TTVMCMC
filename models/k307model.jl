#julia 6 version
#TTVFaster statistical model
#takes B = sigma^(1/2) externally (Lower diagonal)
#takes pmeans externally
#close to Kepler 307
#formerly named "TTVmodel3.jl"
#using ForwardDiff

include("../utils/TTVfunctions.jl")

P_b =10.5
P_c =13.0

t0_b=784.0
t0_c=785.0

k_b= 0.011
k_c= 0.004

h_b= -0.04
h_c= -0.029

#adjust argument of periastron for TTVFaster
e_b=sqrt(k_b^2 +h_b^2)
peri_b= atan2(h_b,k_b)
peri_b-=pi/2

k_b=e_b*cos(peri_b)
h_b=e_b*sin(peri_b)

e_c=sqrt(k_c^2 +h_c^2)
peri_c= atan2(h_c,k_c)
peri_c-=pi/2

k_c=e_c*cos(peri_c)
h_c=e_c*sin(peri_c)

mu_b=8.0 # Me/Msun
mu_c=4.0

mu_b=3.003e-6*mu_b
mu_c=3.003e-6*mu_c

ti_b=t0_b
ti_c=t0_c

pinit=[mu_b,P_b,ti_b,k_b,h_b,mu_c,P_c,ti_c,k_c,h_c]
pguess=[mu_b+2.0e-6,P_b,ti_b,k_b+0.005,h_b+0.002,mu_c-1e-6,P_c,ti_c,k_c-0.004,h_c+0.001]

bData=readdlm("../outputs/TTVFasterbData.txt",',')
cData=readdlm("../outputs/TTVFastercData.txt",',')

np=length(pinit)

# z:transformed parameter array
zinit=to_z(pinit)
zguess=to_z(pguess)

include("TTVmodel.jl")
