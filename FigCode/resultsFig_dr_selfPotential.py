# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import sysUtils as su
import gridUtils as gu
import plotUtils as pu
import fig_meshDensities 
import fig_meshAndPartDens
import time
import numpy as np
import mathUtils as mu 
import matplotlib.pyplot as plt

simNames = [
"simAll_stripped00_smallRadius",
"simAll_stripped00_selfPotential",
"simAll_stripped00_selfPotential_run3",
"simAll_stripped00_selfPotential_run2"
]
labels = [
# r'$M_h = 1.1 \times 10^{8} M_\odot, \, 7 m_{22}, \, f_\mathrm{fdm} = 0.5$',
r'$M_* / M_c = 0$',
r'$M_* / M_c = 0.05$',
r'$M_* / M_c = 0.25$',
r'$M_* / M_c = 0.5$',
r'$\rho_f / 2$',
r'$\sqrt{2} \sigma_\mathrm{dm}$',
r'$\sigma_\mathrm{*} / \sqrt{2}$',
'',
''
]
colors = ['r','g','b','c','m','y']
R_halves = [
  0.3,
  0.3,
  0.3,
  0.3,
  ]
M_stars = [
  1e5,
  1e6,
  5e6,
  1e7
  ]


### NFW profile parameters
meandens = 2.775e+11*0.7**2 * 0.31 * 1e-9 # mean density
# Rhalf = 0.3 # half light radius
# dervived parameters
### virial radius: using the relationship between the virial and half-light radius from https://arxiv.org/abs/1212.2980

# need to adjust the mass so its easier to sim?
# Mvir = (4*np.pi/3)*200*meandens*Rvir**3 # virial mass in solar masses
# con = Rvir / Rs # concentration parameter
# rho0 = Mvir/4/np.pi/Rs**3 /(np.log(1+con)-con/(1+con)) # scale density in solar masses / kpc^3



def PlotActualR(d,fo,i):
	Rs = 2. # scale radius in kpc
	Rvir = R_halves[i]/0.015 
	Mvir = (4*np.pi/3)*200*meandens*Rvir**3 # virial mass in solar masses
	con = Rvir / Rs # concentration parameter
	rho0 = Mvir/4/np.pi/Rs**3 /(np.log(1+con)-con/(1+con)) # scale density in solar masses / kpc^3

	rc = au.CoreRadius(d.m22, Mvir=Mvir)
	M_c = au.CoreMass(d.m22, Mvir=Mvir)
	print(d.simName + " r_c:" + str(rc) )
	print(d.simName + "M_c: " + str(M_c) )

	t = np.load(d.dataDir + "t.npy")
	R_half = np.load(d.dataDir + "R_half.npy")

	if (len(t) > 60):
		t = t[::4]
		R_half = R_half[::4]

	label_ = labels[i] if len(labels[i]) > 0 else simNames[i]

	if i == 0:
		fo.AddLine([-1],[0],mk = 'o', color = 'k', label = 'simulated data')
		fo.AddLine([],[],ls = '--', color = 'k', label = 'predictions')
		fo.AddLine(t, R_half,mk = 'o', color = colors[i], label = label_)
	else:
		fo.AddLine(t, R_half,mk = 'o', color = colors[i], label = label_)


def LoadExtras(d):
	d.LoadExtraParams(('fraction_FDM',))
	d.fraction_FDM = d.extraParams['fraction_FDM']


def PlotPredictedR(d,fo,j,M):
	# TODO: this using algorithm from resultsFig_delta_r_with_phi_rms
	dx = d.L / d.N
	t_data = np.load(d.dataDir + "t.npy")
	R_half_data = np.load(d.dataDir + "R_half.npy")

	R,_,_ = gu.sphrGrid(d.N,d.L)
	psi = d.LoadPsi(0)
	rho = np.sum(np.abs(psi)**2,axis=0)
	print(np.sum(np.abs(psi)**2)*d.dx**3 / d.fraction_FDM)

	r, phi_rms = GetPhiRms(d, R_halves[j])
	rvals, F = GetForce_alt(d, R_halves[j], M)
	F *= 1.2
	dr = rvals[1] - rvals[0]

	F_at_r = np.interp(r, rvals, F)

	Potential_Energy = np.cumsum(F*dr)
	v_sigma = np.sqrt(F*rvals)
	sigma_dm = np.max(v_sigma)
	Kinetic_Energy = 0.5*np.sqrt(F*rvals)**2
	Energy_Total = Kinetic_Energy + Potential_Energy - 0.25*au.G*M / rvals**2

	print(d.m22, 2*np.pi * d.hbar_[0] / sigma_dm**2)
	lam_ = 2*np.pi * d.hbar_[0] / sigma_dm
	rho_max = np.max(rho)
	tau = 2*np.pi * d.hbar_[0] / sigma_dm**2

	t = np.linspace(0, d.Tf, 1024)
	N_int = len(t)
	dt = d.Tf / N_int
	R_half_predicted_sol = np.zeros(N_int)
	R_half_predicted_sol[0] = R_half_data[0]
	E_tot_at_r = np.interp(r, rvals, Energy_Total)

	time0 = time.time()
	for i in range(1,N_int):
		F_at_r_val = np.interp(R_half_predicted_sol[i-1], rvals, F)

		phi_rms_at_r_val = np.interp(R_half_predicted_sol[i-1], r, phi_rms)

		delta_E = 0.1*0.5*phi_rms_at_r_val*dt / tau

		delta_R = delta_E / (F_at_r_val + au.G*M/2. / R_half_predicted_sol[i-1]**2)

		R_half_predicted_sol[i] = R_half_predicted_sol[i-1] + delta_R
		
		su.PrintTimeUpdate(i, N_int,time0)

	fo.AddLine(t, R_half_predicted_sol, color = colors[j], ls= '--')


def GetPhiRms(d, Rhalf):
	Rs = 2. # scale radius in kpc
	Rvir = Rhalf/0.015 
	Mvir = (4*np.pi/3)*200*meandens*Rvir**3 # virial mass in solar masses
	con = Rvir / Rs # concentration parameter
	rho0 = Mvir/4/np.pi/Rs**3 /(np.log(1+con)-con/(1+con)) # scale density in solar masses / kpc^3

	rc = au.CoreRadius(d.m22, Mvir=Mvir)
	M_c = au.CoreMass(d.m22, Mvir=Mvir)
	print(d.simName + " r_c:" + str(rc) )
	print(d.simName + "M_c: " + str(M_c) )

	r = np.linspace(0, 10, 1024)
	dr = r[1] - r[0]
	rho1 = au.Core(r, d.m22, au.CoreRadius(d.m22, Mvir=Mvir))
	rho2 = au.CoredNFW(r, Rs, rho0, d.m22, au.CoreRadius(d.m22, Mvir=Mvir))
	# rho =0.5 * (rho1 + rho2)
	rho = rho2
	M_encl = np.cumsum(4*np.pi*r**2 * dr * rho)
	sigma_ = np.sqrt(au.G * Mvir / Rvir)
	lam_ = d.hbar_[0] / sigma_
	# Phi = M_encl * au.G * lam_ / (r + dr /2.)**2 * d.fraction_FDM
	Phi = M_encl * au.G * lam_ / (r)**2

	return r, Phi.astype('float64')


def GetForce_alt(d, Rhalf):
	Rs = 2. # scale radius in kpc
	Rvir = Rhalf/0.015 
	Mvir = (4*np.pi/3)*200*meandens*Rvir**3 # virial mass in solar masses
	con = Rvir / Rs # concentration parameter
	rho0 = Mvir/4/np.pi/Rs**3 /(np.log(1+con)-con/(1+con)) # scale density in solar masses / kpc^3

	rc = au.CoreRadius(d.m22, Mvir=Mvir)
	M_c = au.CoreMass(d.m22, Mvir=Mvir)
	print(d.simName + " r_c:" + str(rc) )
	print(d.simName + "M_c: " + str(M_c) )

	r = np.linspace(0.01, 10, 1024)
	dr = r[1] - r[0]
	rho = au.CoredNFW(r, Rs, rho0, d.m22, au.CoreRadius(d.m22, Mvir=Mvir))
	M_encl = np.cumsum(4*np.pi*r**2 * dr * rho)
	F = au.G*M_encl / r**2

	return r, F.astype('float64')


def GetForce(d,R,rho, M):
	dx = d.L / d.N

	rvals = np.linspace(dx/2.,d.L/2.,1024)
	F = np.zeros(len(rvals))

	for i in range(len(rvals)):
		r_ = rvals[i]
		M_encl = np.sum(rho[R<r_])*dx**3
		F[i] = au.G * (M_encl/d.fraction_FDM)/ r_**2

	return rvals, F




def PlotPredictedR_alt(d,fo,j, M):
	# TODO: this using algorithm from resultsFig_delta_r_with_phi_rms
	dx = d.L / d.N
	t_data = np.load(d.dataDir + "t.npy")
	R_half_data = np.load(d.dataDir + "R_half.npy")

	R,_,_ = gu.sphrGrid(d.N,d.L)
	psi = d.LoadPsi(0)
	rho = np.sum(np.abs(psi)**2,axis=0)
	print(np.sum(np.abs(psi)**2)*d.dx**3 / d.fraction_FDM)

	r, phi_rms = GetPhiRms(d, R_halves[j])
	# r = np.linspace(dx / 2., d.L/2., len(phi_rms))
	# rvals, F = GetForce(d,R,rho)
	rvals, F = GetForce_alt(d, R_halves[j])
	# print(F[10])
	# F *= 10.
	# print(F[10])
	dr = rvals[1] - rvals[0]

	F_at_r = np.interp(r, rvals, F)

	Potential_Energy = np.cumsum(F*dr)
	v_sigma = np.sqrt(F*rvals)
	sigma_dm = np.max(v_sigma)
	Kinetic_Energy = 0.5*np.sqrt(F*rvals)**2
	Energy_Total = Kinetic_Energy + Potential_Energy - 1.8*au.G*M / rvals

	# print(d.m22, 2*np.pi * d.hbar_[0] / sigma_dm**2)
	lam_ = 2*np.pi * d.hbar_[0] / sigma_dm
	rho_max = np.max(rho)
	tau = 2*np.pi * d.hbar_[0] / sigma_dm**2

	t = np.linspace(0, d.Tf, 512)
	N_int = len(t)
	dt = d.Tf / N_int
	R_half_predicted_sol = np.zeros(N_int)
	R_half_predicted_sol[0] = R_half_data[0]
	E_tot_at_r = np.interp(r, rvals, Energy_Total)

	time0 = time.time()
	for i in range(1,N_int):
		E_tot_at_r_val = np.interp(R_half_predicted_sol[i-1], rvals, Energy_Total)
		# print(E_tot_at_r_val, E_tot_at_r)
		phi_rms_at_r_val = np.interp(R_half_predicted_sol[i-1], r, phi_rms)
		E_final = 0.2*0.5*phi_rms_at_r_val*dt / tau + E_tot_at_r_val
		R_final = np.interp(E_final,E_tot_at_r,r)
		# print(phi_rms_at_r_val,E_tot_at_r_val, dt / tau, R_final - R_half_predicted_sol[i-1])
		# print(R_final - R_half_predicted_sol[i-1], R_final)
		R_half_predicted_sol[i] = R_final
		
		su.PrintTimeUpdate(i, N_int,time0)

	fo.AddLine(t, R_half_predicted_sol, color = colors[j], ls= '--')

	alpha = 1.8
	beta = .2*.5 / np.pi / 2 * d.fraction_FDM
	R_i = R_half_predicted_sol[0]
	C = alpha*au.G*M / R_i - 2*R_i / beta / sigma_dm
	print(d.fraction_FDM)

	a = -2/ beta / sigma_dm
	b = t-C
	c = alpha*au.G*M

	predict_plus = -b/2/a + np.sqrt(b**2 - 4*a*c)/2/a
	predict_minus = -b/2/a - np.sqrt(b**2 - 4*a*c)/2/a
	print(b**2, C, 2*R_i / beta / sigma_dm)
	predict_massless = sigma_dm*.2*.5/4. / np.pi  * t * d.fraction_FDM + R_i
	fo.AddLine(t, predict_plus, color = colors[j], ls= ':', alpha = .3)



def Main():
	fo = pu.FigObj()

	ax = None
	for i in range(len(simNames)):
		name = simNames[i]
		d = do.MeshDataObj(name)
		d0 = do.MeshDataObj(simNames[0])
		LoadExtras(d)
		LoadExtras(d0)
		PlotActualR(d,fo,i)
		PlotPredictedR_alt(d0,fo,i, M = M_stars[i])

	plt.rc('legend',fontsize=16) # using a size in points
	fo.legend()

	fo.SetXLabel(r'$t \, [\mathrm{Myr}]$')
	fo.SetYLabel(r'$R \, [\mathrm{kpc}]$')

	fo.SetYLim(0,2.5)
	fo.SetXLim(0,9000.0)

	fo.SetTitle("Heating in halo")
	fo.legend()

	fo.Save("solitonHeatingHalo_selfPotential")

	fo.show()

if __name__ == "__main__":
	Main()