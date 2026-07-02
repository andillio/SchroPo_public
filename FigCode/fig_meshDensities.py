# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import matplotlib.pyplot as plt 
import numpy as np 

# simName = "ics_2_5m22"
# simName = "solitonData_light"
simName = "haloTest"


def GetCenterOfMass(rho):
	pass


def PlotStuff(d):
	initial_drop = 0
	mid_drop = d.data_drops // 2
	final_drop = d.data_drops 

	data_drops = d.data_drops

	N = d.N
	T = d.Tf
	L = d.L 
	dx = L/N
	x = [-L/2., L/2.]

	fo = pu.FigObj(3,2)

	psi = d.LoadPsi(initial_drop)
	d.RollSim()
	print(np.sum(np.abs(psi)**2)*dx**3)
	slice_ = d.GetFieldDensity()[N//2,:,:]
	fo.AddDens2d(x, np.log10(slice_) )
	T_i = T * float(initial_drop) / data_drops
	fo.AddText(r'$%i \, [\mathrm{Myr}]$'%(T_i))
	fo.RemoveXLabels()

	psi = d.LoadPsi(mid_drop)
	d.RollSim()
	slice_ = d.GetFieldDensity()[N//2,:,:]
	fo.AddDens2d(x, np.log10(slice_) )
	T_m = T * float(mid_drop) / data_drops
	fo.AddText(r'$%i \, [\mathrm{Myr}]$'%(T_m))
	fo.RemoveXLabels()
	fo.RemoveYLabels()	
	fo.SetTitle(r"ULDM halo density")


	psi = d.LoadPsi(final_drop)
	d.RollSim()
	slice_ = d.GetFieldDensity()[N//2,:,:]
	ax, im1 = fo.AddDens2d(x, np.log10(slice_) )
	T_f = T * float(final_drop) / data_drops
	fo.AddText(r'$%i \, [\mathrm{Myr}]$'%(T_f))
	fo.RemoveXLabels()
	fo.RemoveYLabels()


	# fo.AddTextRightLabel(r'Density slices')

	psi = d.LoadPsi(initial_drop)
	proj = np.sum( d.GetFieldDensity() , axis = 0)*dx
	fo.AddDens2d(x, np.log10(proj))

	psi = d.LoadPsi(mid_drop)
	proj = np.sum( d.GetFieldDensity() , axis = 0)*dx
	fo.AddDens2d(x, np.log10(proj))
	fo.RemoveYLabels()

	psi = d.LoadPsi(final_drop)
	proj = np.sum( d.GetFieldDensity() , axis = 0)*dx
	ax, im2 = fo.AddDens2d(x, np.log10(proj))
	fo.RemoveYLabels()
	# fo.AddColorbar(im2)
	# fo.AddTextRightLabel(r'Density projections')

	import matplotlib
	cax = fo.fig.add_axes([.9, 0.11+.385, 0.01, 0.385])
	cbar = fo.fig.colorbar(im1, cax=cax, orientation='vertical',  norm=matplotlib.colors.LogNorm())
	cbar.ax.set_yticks([0,2,4,6])
	cbar.ax.set_yticklabels([r'$10^0$',r'$10^2$',r'$10^4$',r'$10^6$'])
	cbar.set_label(r'denisty slice $[\mathrm{M_\odot / kpc^3}]$', rotation=270, labelpad=30)
	cax = fo.fig.add_axes([.9, 0.11, 0.01, 0.385])
	cbar = fo.fig.colorbar(im2, cax=cax, orientation='vertical',  norm=matplotlib.colors.LogNorm())
	cbar.ax.set_yticks([5,6,7])
	cbar.ax.set_yticklabels([r'$10^5$',r'$10^6$',r'$10^7$'])
	cbar.set_label(r'projected denisty $[\mathrm{M_\odot / kpc^3}]$', rotation=270, labelpad=30)


	fo.RemoveWhiteSpace()

	fo.save(d.dataDir + 'densities')


def Main(name):
	d = do.MeshDataObj(name)
	PlotStuff(d)


if __name__ == "__main__":
	Main(simName)
	plt.show()