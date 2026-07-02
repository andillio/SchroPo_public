# pylint: disable=C,W
# this test is designed to check that our distribution
# is approximately stable under an external nfw potential
import numpy as np_
try:
	import cupy as np
except ImportError:
	import numpy as np
import astroUtils as au
import gridUtils as gu
import sys
sys.path.insert(1, 'Solvers')
import Solvers.meshSolver as MS
import os

# test is ready to run
# mestTest2 - control test
# meshTest2b - use new update rule
simName = "haloTest"
N = 32
data_drops = 150
padded = True
cf = .1
nf = 1
C = au.G*4*np.pi
Tf = 1500.
gpu = False
fraction_FDM = .5


meandens = 2.775e+11*0.7**2 * 0.31 * 1e-9 # mean density
Rhalf = 0.3 # half light radius
# dervived parameters
### virial radius: using the relationship between the virial and half-light radius from https://arxiv.org/abs/1212.2980
Rs = 2. # scale radius in kpc
Rvir = Rhalf/0.015 
# need to adjust the mass so its easier to sim?
Mvir = (4*np.pi/3)*200*meandens*Rvir**3 # virial mass in solar masses
con = Rvir / Rs # concentration parameter

cp = np_
if gpu:
	cp = np
L = Rvir*2 / np.sqrt(3) # box size in kpc
m22 = cp.array([5.0])
rho0 = Mvir/4/cp.pi/Rs**3 /(cp.log(1+con)-con/(1+con)) # scale density in solar masses / kpc^3


def V_ext_func(R):
	return au.V_NFW(R, Rs, rho0)


def GetExternalPotential():
	R, _, _ = gu.sphrGrid(N, L, gpu = gpu)
	return V_ext_func(R)*(1. - fraction_FDM)


def SetICs():
	cp = np_
	if gpu:
		cp = np
	s = MS.Solver()
	### set simulation parameters
	s.SetParams(simName=simName, N = N, data_drops = data_drops, padded=padded,
	 cf=cf, L = L, m22 = m22, C = C, Tf = Tf, gpu = gpu)
	### set initial field

	s.initial_drop = 0
	s.T_initial = 0.

	psi = cp.zeros((nf,N,N,N)) + 0j

	IC_dir = os.path.join(os.getcwd(), "../HaloConstructor_public/ICs")
	psi[0,:,:,:] = cp.load(os.path.join(IC_dir, 'eri_300Emax.npy')) * cp.sqrt(fraction_FDM)

	s.V_ext = GetExternalPotential()
	s.set_psi(psi)
	s.set_K()
	s.oldUpdateRule = True

	return s


if __name__ == "__main__":
	# set up sim ics
	s = SetICs()

	# s.ShowDensity("initial_density")
	# s.Update(10.)
	# s.ShowDensity()
	# assert(0)

	# - run sim
	s.RunSim()
	# s.ShowDensity("final_density")
	