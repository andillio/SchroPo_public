# pylint: disable=C,W
# this test is designed to check that our distribution is stable in
# isolation: zero external NFW potential, zero background FDM field.
# gravity for the stars comes only from their mutual N-body forces.


try:
	import cupy as np
except ImportError:
	import numpy as np
import numpy as np_
import astroUtils as au
import gridUtils as gu
import mathUtils as mu
import sysUtils as su
import os
import glob
import re
import sys
sys.path.insert(1, 'Solvers')
import Solvers.mesh_solver_vect as MS

# test is ready to run
# mestTest2 - control test
# meshTest2b - use new update rule
simName = "plummer_isolated_256_run2"
Checkpoint = True
checkpoint_dir = f'/home/jdarne1/Eberhardt_Massive_Stars/SchroPo_public/Data/{simName}'

N = 256
data_drops = 40
padded = True
cf = .1 / 3.
nf = 1
C = au.G*4*np.pi
Tf = 1000.
gpu = True
fraction_FDM = 0. # zero FDM background for this test
n_stars = int(3e4)

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


def StarICs_Plummer():
	cp = np_
	if gpu:
		cp = np

	cp.random.seed(42)

	X1 = cp.random.uniform(0, 1, n_stars)
	X2 = cp.random.uniform(0, 1, n_stars)
	X3 = cp.random.uniform(0, 1, n_stars)

	a = a_stars * (X1**(-2/3) - 1)**(-0.5)

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

	# m_star = M_stars / n_stars

	# W = -cp.sum(m_star * r_mag * au.G*M_stars * r_mag / (r_mag**2 + a_stars**2)**(3/2))

	# KE = 0.5 * cp.sum(m_star * v_mag**2)

	# scale_factor = cp.sqrt(cp.abs(W)/(2*KE))

	# v_mag *= scale_factor
	# v *= scale_factor


	return r, v


def get_latest_drop(checkpoint_dir):
	# psi is all-zero and frozen here, so only r and v need restoring

	drops = {}

	for path in glob.glob(os.path.join(checkpoint_dir, 'r', 'drop*.npy')):
		m = re.fullmatch(r'drop(\d+)\.npy', os.path.basename(path))
		if m:
			drops[int(m.group(1))] = path

	for i in sorted(drops, reverse=True):
		v_path = os.path.join(checkpoint_dir, 'v', f'drop{i}.npy')
		if os.path.isfile(v_path):
			return i, drops[i], v_path

	return None


def SetICs():
	cp = np_
	if gpu:
		cp = np

	initial_drop = 0
	T_initial = 0.
	Tf_run = Tf
	drops_run = data_drops

	found = get_latest_drop(checkpoint_dir) if Checkpoint else None

	if found is not None:
		initial_drop, r_path, v_path = found
		if initial_drop >= data_drops:
			raise RuntimeError(f"checkpoint at drop {initial_drop}: sim already complete")
		r = cp.load(r_path)
		v = cp.load(v_path)
		T_initial = Tf * initial_drop / data_drops
		Tf_run = Tf - T_initial
		drops_run = data_drops - initial_drop
		print(f"resuming from drop {initial_drop} (T = {T_initial:.1f})")
	else:
		if Checkpoint:
			print("no checkpoint found, generating new ICs")
		r,v = StarICs_Plummer()

	mp = cp.ones(n_stars)*M_stars / n_stars

	s = MS.Solver()
	### set simulation parameters
	s.SetParams(simName=simName, N = N, data_drops = drops_run, padded=padded,
	 cf=cf, L = L, m22 = m22, C = C, Tf = Tf_run, r = r, v = v,
	 np = n_stars, mp = mp, gpu = gpu)
	### set initial field
	s.D = 3
	# no s.M_encl_func / s.M_encl_func_on: zero external NFW potential
	s.explicit_particle_forces = True
	s.eps = 1e-5 / 3.
	s.initial_drop = initial_drop
	s.T_initial = T_initial

	# zero FDM field: psi stays all-zero and frozen for the whole run
	psi = cp.zeros((nf,N,N,N)) + 0j

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
	