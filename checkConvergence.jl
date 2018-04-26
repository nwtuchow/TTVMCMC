#see if chains converged
using Plots

pdata=readdlm("../../../Documents/Exoplanet_ttv_data/params_kepler307b.csv")

pdata=transpose(pdata)
parray1= pdata[500001:2000000,:]
parray2= pdata[2000001:3500000,:]
parray3= pdata[3500001:end,:]

plotly()

totmassplot=histogram2d(pdata[:,1]/3.003e-6,pdata[:,6]/3.003e-6,
            xlabel="Planet b mass ratio",
            ylabel="Planet c mass ratio",
            title= "Full Dataset",
            nbins=40,
            fc=:plasma)

massplot1=histogram2d(parray1[:,1]/3.003e-6,parray1[:,6]/3.003e-6,
            xlabel="Planet b mass ratio",
            ylabel="Planet c mass ratio",
            nbins=40,
            fc=:plasma)

massplot2=histogram2d(parray2[:,1]/3.003e-6,parray2[:,6]/3.003e-6,
            xlabel="Planet b mass ratio",
            ylabel="Planet c mass ratio",
            nbins=40,
            fc=:plasma)

massplot3=histogram2d(parray3[:,1]/3.003e-6,parray3[:,6]/3.003e-6,
            xlabel="Planet b mass ratio",
            ylabel="Planet c mass ratio",
            nbins=40,
            fc=:plasma)

plot(totmassplot,massplot1,massplot2,massplot3, layout=4)

totkplot=histogram2d(pdata[:,4],pdata[:,9],
            xlabel="k_b",
            ylabel="k_c",
            title="Full Dataset",
            nbins=40,
            fc=:plasma)



kplot1=histogram2d(parray1[:,4],parray1[:,9],
            xlabel="k_b",
            ylabel="k_c",
            nbins=40,
            fc=:plasma)

kplot2=histogram2d(parray2[:,4],parray2[:,9],
            xlabel="k_b",
            ylabel="k_c",
            nbins=40,
            fc=:plasma)

kplot3=histogram2d(parray3[:,4],parray3[:,9],
            xlabel="k_b",
            ylabel="k_c",
            nbins=40,
            fc=:plasma)

plot(totkplot,kplot1,kplot2,kplot3, layout=4)

tothplot=histogram2d(pdata[:,5],pdata[:,10],
            xlabel="h_b",
            ylabel="h_c",
            title="Full Dataset",
            nbins=40,
            fc=:plasma)


hplot1=histogram2d(parray1[:,5],parray1[:,10],
            xlabel="h_b",
            ylabel="h_c",
            nbins=40,
            fc=:plasma)

hplot2=histogram2d(parray2[:,5],parray2[:,10],
            xlabel="h_b",
            ylabel="h_c",
            nbins=40,
            fc=:plasma)

hplot3=histogram2d(parray3[:,5],parray3[:,10],
            xlabel="h_b",
            ylabel="h_c",
            nbins=40,
            fc=:plasma)

plot(tothplot,hplot1,hplot2,hplot3, layout=4)
