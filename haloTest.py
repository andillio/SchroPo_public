# pylint: disable=C,W
# this test is designed to check that our distribution
# is approximately stable under an external nfw potential
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
simName = "haloTest_w_plummer"
N = 256
data_drops = 150
padded = True
cf = .1
nf = 1
C = au.G*4*np.pi
Tf = 1500.
gpu = True
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


M_stars = 1e6

a_stars = 0.1 # scale radius of stars in kpc

def V_Plummer(R):
	return -au.G*M_stars / cp.sqrt(R**2 + a_stars**2)

def rho_Plummer(r):
	return 3*M_stars/(4*cp.pi*a_stars**3) * (1 + r**2/a_stars**2)**(-5/2)

def M_Plummer(r):
	return M_stars * r**3 / (r**2 + a_stars**2)**(3/2)



def V_ext_func(R):
	return au.V_NFW(R, Rs, rho0)*(1. - fraction_FDM) + V_Plummer(R)

# function used to calculate cdm effect on stars
def M_encl_func(r):
	return au.M_NFW(r, Rs, rho0)*(1. - fraction_FDM) + M_Plummer(r)



central_dens_plummer = rho_Plummer(0.)
print(f"central density of plummer profile: {central_dens_plummer:.3e} Msun/kpc^3")

print(f"Mass of plummer sphere: {M_Plummer(20.):.3e} Msun within 20 kpc")

print(f"virial mass of halo: {Mvir:.3e} Msun within {Rvir:.3e} kpc")


def SetICs():
	cp = np_
	if gpu:
		cp = np
	s = MS.Solver()
	### set simulation parameters
	s.SetParams(simName=simName, N = N, data_drops = data_drops, padded=padded,
	 cf=cf, L = L, m22 = m22, C = C, Tf = Tf, gpu = gpu)
	### set initial field
	s.D = 3
	s.M_encl_func = M_encl_func
	s.V_ext_mesh_func = V_ext_func
	s.M_encl_func_on = True
	s.initial_drop = 0
	s.T_initial = 0.

	psi = cp.zeros((nf,N,N,N)) + 0j

	IC_dir = os.path.join(os.getcwd(), "../HaloConstructor_public/Code")
	psi[0,:,:,:] = cp.load(os.path.join(IC_dir, 'eri_200Emax_256.npy')) * cp.sqrt(fraction_FDM)

	s.set_psi(psi)
	
	field_dens = s.GetFieldDensity()
	central_soliton_dens = cp.max(field_dens)
	print(f"central soliton density: {central_soliton_dens:.3e} Msun/kpc^3")

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
	s.ShowDensity("final_density")

	import matplotlib.pyplot as plt

	to_cpu = lambda x: np_.asarray(x.get()) if hasattr(x, 'get') else np_.asarray(x)

	r_grid   = to_cpu(s.R).ravel()              
	rho_psi  = to_cpu(s.GetFieldDensity()).ravel()


	r_min = float(r_grid.min())      # central pixel: sqrt(3)/2 * dx
	r_max = float(to_cpu(L)) / 2.    # last radius with full spherical coverage
	bins  = np_.logspace(np_.log10(r_min), np_.log10(r_max), 40)

	which        = np_.digitize(r_grid, bins)
	r_cent       = np_.sqrt(bins[:-1] * bins[1:])          
	rho_psi_prof = np_.array([rho_psi[which == i].mean() if np_.any(which == i)
							else np_.nan
							for i in range(1, len(bins))])


	r_fine = np_.logspace(np_.log10(0.02), np_.log10(r_max), 300)
	rho_plum_fine = to_cpu(rho_Plummer(np_.asarray(r_fine)))

	rho_plum = np_.array([rho_Plummer(r_cent[i]) for i in range(len(r_cent))])

	plt.figure(figsize=(6, 5))
	plt.loglog(r_cent, rho_psi_prof, "o-", label=r"$|\psi|^2$ (field)")
	plt.loglog(r_cent, rho_plum,      "--", label="Plummer (stars)")
	plt.axvline(a_stars, color="grey", ls=":", lw=1, label=r"$a_\star$")
	plt.xlabel(r"$r$ [kpc]")
	plt.ylabel(r"$\rho$ [$M_\odot\,\mathrm{kpc}^{-3}$]")
	plt.legend()
	plt.tight_layout()
	plt.savefig(f"/home/joshua/PhD_year_1/Eberhardt_Massive_Stars/SchroPo_public/Data/haloTest_w_plummer/{simName}_density_profile.png", dpi=150)
	plt.show()
		