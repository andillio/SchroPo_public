# pylint: disable=C,W
# this test is designed to check that our distribution
# is approximately stable under an external nfw potential

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
simName = "starTest_Plummer"
N = 256
data_drops = 20
padded = True
cf = .1
nf = 1
C = au.G*4*np.pi
Tf = 1500.
gpu = True
fraction_FDM = .5
n_stars = int(1e4)

# Plummer sphere for the sampled stars — matches haloTest.py
M_stars = 1e6   # total Plummer mass in solar masses
a_stars = 0.1   # Plummer scale radius in kpc

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

# function used to calculate cdm effect on stars
def M_encl_func(r):
	return au.M_NFW(r, Rs, rho0)*(1. - fraction_FDM)

def GetExternalPotential():
	R, _, _ = gu.sphrGrid(N, L, gpu = gpu)
	return V_ext_func(R)*(1. - fraction_FDM)


def M_Plummer(r):
	return M_stars * r**3 / (r**2 + a_stars**2)**(3/2)

def V_Plummer(R):
	return -au.G*M_stars / cp.sqrt(R**2 + a_stars**2)


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
	r_hat = su.gpuThis(r_hat)
	v_vec = cp.random.normal(0, R_initial_star, size = (n_stars,3))
	V_mag = np.sqrt(M_Scale * au.G / R_mag) * 1e-6

	for i in range(n_stars):
		r[i,:] = R_mag[i]*r_hat[i]
		v[i,:] = V_mag[i]*v_vec[i]

	return r, v

def StarICs_Plummer():
	cp = np_
	if gpu:
		cp = np

	X1 = cp.random.uniform(0, 1, n_stars)
	X2 = cp.random.uniform(0, 1, n_stars)
	X3 = cp.random.uniform(0, 1, n_stars)

	a = (X1**(-2/3) - 1)**(-0.5)

	z = (1 - 2*X2) * a
	x = cp.sqrt(a**2 - z**2) * cp.cos(2*cp.pi*X3)
	y = cp.sqrt(a**2 - z**2) * cp.sin(2*cp.pi*X3)

	v_esc = cp.sqrt(2*au.G*M_stars / a_stars)*(1+a)**(-1/4)

	v = []

	for star in range(n_stars):
		X4 = cp.random.uniform(0, 1)
		X5 = cp.random.uniform(0, 1)
		while X4**2 * (1 - X4**2)**3.5 <= 0.1*X5:
			X4 = cp.random.uniform(0, 1)
			X5 = cp.random.uniform(0, 1)
		
		q = X4

		v_mag = v_esc[star] * q

		X6 = cp.random.uniform(0, 1)
		X7 = cp.random.uniform(0, 1)

		v_z = (1 - 2*X6) * v_mag
		v_x = cp.sqrt(v_mag**2 - v_z**2) * cp.cos(2*cp.pi*X7)
		v_y = cp.sqrt(v_mag**2 - v_z**2) * cp.sin(2*cp.pi*X7)

		v.append([v_x, v_y, v_z])
	
	r = cp.array([x, y, z]).T
	r_mag = cp.sqrt(cp.sum(r**2, axis = 1))
	v = cp.array(v)
	v_mag = cp.sqrt(cp.sum(v**2, axis = 1))

	#Virial check

	m_star = M_stars / n_stars

	W = -cp.sum(m_star * r_mag * au.G*M_stars * r_mag / (r_mag**2 + a_stars**2)**(3/2))

	KE = 0.5 * cp.sum(m_star * v_mag**2)

	scale_factor = cp.sqrt(cp.abs(W)/(2*KE))

	v_mag *= scale_factor
	v *= scale_factor


	return r, v


def SetICs():
	cp = np_
	if gpu:
		cp = np

	r,v = StarICs_Plummer()

	s = MS.Solver()
	### set simulation parameters
	s.SetParams(simName=simName, N = N, data_drops = data_drops, padded=padded,
	 cf=cf, L = L, m22 = m22, C = C, Tf = Tf, r = r, v = v,
	 np = n_stars, mp = 0, gpu = gpu)
	### set initial field
	s.D = 3
	s.M_encl_func = M_encl_func
	s.V_ext_mesh_func = V_ext_func
	s.M_encl_func_on = True
	s.explicit_particle_forces = True
	s.eps = 1e-5
	s.initial_drop = 0
	s.T_initial = 0.

	psi = cp.zeros((nf,N,N,N)) + 0j
	
	psi[0,:,:,:] = cp.load('Data/haloTest_w_plummer/psi/drop150.npy') * cp.sqrt(fraction_FDM)

	#s.V_ext = GetExternalPotential()
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
	