# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import matplotlib.pyplot as plt 
import numpy as np 

simName = 'physTest1a_stripped04'


def PlotStuff(d):
	data_drops = d.data_drops
	
	initial_drop = 0 
	mid_drop = d.data_drops // 2
	final_drop = d.data_drops 

	N = d.N
	T = d.Tf
	L = d.L 
	dx = L/N
	x = [-L/2., L/2.]
	print(dx)

	fo = pu.FigObj(3,2)

	psi = d.LoadPsi(initial_drop)
	slice_ = d.GetFieldDensity()[N//2,:,:]
	fo.AddDens2d(x, np.log(slice_) )
	T_i = T * float(initial_drop) / data_drops
	fo.AddText(r'$%i \, [\mathrm{Myr}]$'%(T_i))
	fo.RemoveXLabels()

	psi = d.LoadPsi(mid_drop)
	slice_ = d.GetFieldDensity()[N//2,:,:]
	fo.AddDens2d(x, np.log(slice_) )
	T_m = T * float(mid_drop) / data_drops
	fo.AddText(r'$%i \, [\mathrm{Myr}]$'%(T_m))
	fo.RemoveXLabels()
	fo.RemoveYLabels()	

	psi = d.LoadPsi(final_drop)
	slice_ = d.GetFieldDensity()[N//2,:,:]
	fo.AddDens2d(x, np.log(slice_) )
	T_f = T * float(final_drop) / data_drops
	fo.AddText(r'$%i \, [\mathrm{Myr}]$'%(T_f))
	fo.RemoveXLabels()
	fo.RemoveYLabels()
	fo.AddTextRightLabel(r'Density slices')

	r,v,psi = d.LoadData(initial_drop,center = False)
	print(np.mean(r, axis = 0))
	proj = np.sum( d.GetFieldDensity() , axis = 0)*dx
	rX = r[:,2]
	rY = r[:,1]
	fo.AddDens2d(x, np.log(proj))
	fo.AddLine(rX, rY, ls = ' ', mk = '.', alpha = 0.01, color = 'r')
	fo.SetXLim(-L/2, L/2)
	fo.SetYLim(-L/2, L/2)

	r,v,psi = d.LoadData(mid_drop,center = False)
	print(np.mean(r, axis = 0))
	proj = np.sum( d.GetFieldDensity() , axis = 0)*dx
	rX = r[:,2]
	rY = r[:,1]
	proj = np.sum( d.GetFieldDensity() , axis = 0)*dx
	fo.AddDens2d(x, np.log(proj))
	fo.AddLine(rX, rY, ls = ' ', mk = '.', alpha = 0.01, color = 'r')
	fo.SetXLim(-L/2, L/2)
	fo.SetYLim(-L/2, L/2)
	fo.RemoveYLabels()


	r,v,psi = d.LoadData(final_drop,center = False)
	print(np.mean(r, axis = 0))
	proj = np.sum( d.GetFieldDensity() , axis = 0)*dx
	rX = r[:,2]
	rY = r[:,1]
	proj = np.sum( d.GetFieldDensity() , axis = 0)*dx
	fo.AddDens2d(x, np.log(proj))
	fo.AddLine(rX, rY, ls = ' ', mk = '.', alpha = 0.01, color = 'r')
	fo.SetXLim(-L/2, L/2)
	fo.SetYLim(-L/2, L/2)
	fo.RemoveYLabels()
	fo.AddTextRightLabel(r'Density projections')

	fo.RemoveWhiteSpace()

	fo.save(d.dataDir + 'densitiesAndPart')


def Main(name):
	d = do.MeshDataObj(name)
	PlotStuff(d)


if __name__ == "__main__":
	Main(simName)
	plt.show()