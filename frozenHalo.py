# pylint: disable=C,W
# this test is designed to check that our distribution
# is approximately stable under an external nfw potential
try:
	import cupy as np
except ImportError:
	import numpy as np
import numpy as np_
import astroUtils as au
import gridUtils as gu
import mathUtils as mu
import sysUtils as su
import sys
sys.path.insert(1, 'Solvers')
import Solvers.meshSolver as MS

# test is ready to run
# mestTest2 - control test
# meshTest2b - use new update rule
simName = "starTest"
N = 32
data_drops = 20
padded = True
cf = .1
nf = 1
C = au.G*4*np.pi
Tf = 1500.
gpu = False
fraction_FDM = .5
n_stars = int(1e4)

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
L = Rvir*2 / cp.sqrt(3)
R_initial_star = 2.
m22 = cp.array([5.0])
rho0 = Mvir/4/cp.pi/Rs**3 /(cp.log(1+con)-con/(1+con)) # scale density in solar masses / kpc^3


def V_ext_func(R):
	return au.V_NFW(R, Rs, rho0)


def GetExternalPotential():
	R, _, _ = gu.sphrGrid(N, L, gpu = gpu)
	return V_ext_func(R)*(1. - fraction_FDM)


def StarICs():
	cp = np_
	if gpu:
		cp = np	
	
	rng = cp.random.default_rng()
	M_Scale = rho0*L**3

	r = cp.zeros((n_stars, 3))
	v = cp.zeros((n_stars, 3))

	R_mag = cp.abs(cp.random.normal(0, R_initial_star, size = (n_stars)))
	r_hat = mu.random_unit_vectors(n_stars)
	#r_hat = su.gpuThis(r_hat)
	v_vec = cp.random.normal(0, R_initial_star, size = (n_stars,3))
	V_mag = np.sqrt(M_Scale * au.G / R_mag) * 1e-6

	for i in range(n_stars):
		r[i,:] = R_mag[i]*r_hat[i]
		v[i,:] = V_mag[i]*v_vec[i]

	return r, v


def SetICs():
	cp = np_
	if gpu:
		cp = np

	r,v = StarICs()

	s = MS.Solver()
	### set simulation parameters
	s.SetParams(simName=simName, N = N, data_drops = data_drops, padded=padded,
	 cf=cf, L = L, m22 = m22, C = C, Tf = Tf, r = r, v = v,
	 np = n_stars, mp = 0, gpu = gpu)
	### set initial field

	s.initial_drop = 0
	s.T_initial = 0.

	psi = cp.zeros((nf,N,N,N)) + 0j
	
	psi[0,:,:,:] = cp.load('Data/haloTest/psi/drop150.npy') * cp.sqrt(fraction_FDM)

	s.V_ext = GetExternalPotential()
	s.set_psi(psi)
	s.set_K()
	s.oldUpdateRule = True
	s.freezeDensity = True

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
	s.ShowDensity("frozen_final_density")
	