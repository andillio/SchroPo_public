# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import numpy as np 

simNames = [
 "test_positive_self_interact",
]
colors = ['r','b','g','c','m','y','k']


def PlotHalfLightRadius(d, fo, j):
	data_drops = d.data_drops

	t = np.linspace(0, d.Tf, data_drops + 1)
	R_half = np.zeros(data_drops + 1)

	for i in range(data_drops + 1):
		r, v = d.LoadCorpData(i)
		R = np.sqrt(np.sum(np.abs(r)**2, axis = 1))
		R_half[i] = np.median(R)

	if j == 0:
		fo.AddPlot(t, R_half, color = colors[j])
	else:
		fo.AddLine(t, R_half, color = colors[j])



if __name__ == "__main__":
	fo = pu.FigObj()

	for i in range(len(simNames)):
		d = do.MeshDataObj(simNames[i])
		PlotHalfLightRadius(d, fo, i)

	fo.SetYLabel(r'$R$')
	fo.SetXLabel(r'$t$')
	fo.show()