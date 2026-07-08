# pylint: disable=C,W
# simulates stars + fdm + cdm
# represented as corpuscular particles, a field,
# and an external potential respectively

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import numpy as np_
try:
	import cupy as np
except ImportError:
	import numpy as np
import astroUtils as au
import gridUtils as gu
import plotUtils as pu
import sys
sys.path.insert(1, 'Solvers')
import Solvers.mesh_solver_vect as MS

# sims
# - physTest1a: ics1a, fraction 0.5
# - physTest1b: ics1a, fraction 0.5
# - need to test physTest1a with old update
simName = "physTest1a_plummer"
N = 256
data_drops = 10
padded = True
cf = .1
nf = 1
C = au.G*4*np.pi
Tf = 2000.
gpu = True
fraction_FDM = 0.5

### NFW profile parameters
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
dx = L/N
m22 = cp.array([5.0])
rho0 = Mvir/4/cp.pi/Rs**3 /(cp.log(1+con)-con/(1+con)) # scale density in solar masses / kpc^3
rho0_ = Mvir/4/np_.pi/Rs**3 /(np_.log(1+con)-con/(1+con)) # scale density in solar masses / kpc^3

### EDIT ME
### IC info
# field_dir = 'Data/physTest1a/psi/drop10.npy'
# r_dir = 'Data/physTest1a/r/drop10.npy'
# v_dir = 'Data/physTest1a/v/drop10.npy'
r_dir              = 'Data/starTest_Plummer/r/drop20.npy'
v_dir              = 'Data/starTest_Plummer/v/drop20.npy'
field_dir          = 'Data/starTest_Plummer/psi/drop20.npy'

M_stars = 1e6 # plummer sphere mass
a_stars = 0.1

initial_drop = 0
initial_time = 0.

hbar_ = au.h_tilde(m22)[0]
sigma = cp.sqrt(au.G * Mvir / Rvir)
lam = hbar_ / sigma
print(lam / sigma)

# function used to calculate cdm effect on fdm
def V_ext_func(R):
	return au.V_NFW(R, Rs, rho0)*(1. - fraction_FDM)

# function used to calculate cdm effect on stars
def M_encl_func(r):
	return au.M_NFW(r, Rs, rho0)*(1. - fraction_FDM)


def SetICs():

	cp = np_
	if gpu:
		cp = np

	psi = cp.zeros((nf,N,N,N)) + 0j
	r = cp.load(r_dir)
	r %= L
	r[r < -1*L/2.] += L
	r[r > L/2.] -= L
	v = cp.load(v_dir)
	n_p = len(v)
	psi[0,:,:,:] = cp.load(field_dir)

	s = MS.Solver()
	### set simulation parameters
	s.SetParams(simName=simName, N = N, data_drops = data_drops, padded=padded,
	 cf=cf, L = L, m22 = m22, C = C, Tf = Tf, gpu = gpu, r = r, v = v, np = n_p,
	 mp = M_stars/n_p)
	### set initial field
	s.D = 3
	s.M_encl_func = M_encl_func # used for particle forces
	s.V_ext_mesh_func = V_ext_func # used for field forces
	s.M_encl_func_on = True
	s.explicit_particle_forces = True
	s.eps = 1e-9
	s.initial_drop = initial_drop
	s.T_initial = initial_time
	
	s.set_psi(psi)
	s.set_K()

	extras = {}
	extras['fraction_FDM'] = fraction_FDM
	extras['r_ic_directory'] = r_dir
	extras['v_ic_directory'] = v_dir
	extras['field_ic_directory'] = field_dir
	extras['R'] = a_stars
	extras['M'] = M_stars
	extras['Rs'] = Rs
	extras['rho0'] = rho0_
	s.extras = extras
	
	s.oldUpdateRule = True

	s.R,_,_ = gu.sphrGrid(N,L,gpu = gpu)

	return s


if __name__ == "__main__":
	# set up sim ics
	s = SetICs()
	# - run sim
	s.RunSim()