# pylint: disable=C,W
import numpy as np_
import os
import time
import sysUtils as su
import mathUtils as mu
import plotUtils as pu
import types
import baseSolver as BS
CUPY_IMPORTED = True
import warnings as warn 
try:
	import cupy as cp 
except ImportError:
	CUPY_IMPORTED = False


class Solver(BS.Solver):

	def __init__(self, simName = None, N = None, np = 0, data_drops = None, padded = False,
		cf = None, L = None, dx = None, nf = None, m22 = [], hbar_ = [], 
		D = None, C = None, Tf = None, r = [], v = [], dt = None, gpu = False, 
		psi = [], K = [], mp = None):
		"""
		initialize solver object

		:simName: string, simulation data directory name
		:N: int, sim resolution
		:np: int, number of corpuscular particles
		:mp: float, mass of corpuscular particles
		:data_drops: int, number of data outputs
		:padded: bool, pad the density
		:cf: float, courant factor 
		:L: float, box length
		:dx: float, pixel size
		:nf: int, number of fields
		:m22: array-like, [nf] dark matter mass [1e-22 eV/C^2]
		:hbar_: array-like, [nf] hbar/m
		:D: int, number of spatial dimensions
		:C: float, Poisson's constant
		:Tf: float, final sim time
		:r: array-like, [np, D] particle positions
		:v: array-like, [np, D] particle velocities
		:dt: float, initial timestep
		:gpu: bool, run on the gpu
		:psi: array-like, [nf,N^D] spatial field
		:K: array-like, [N^3] kinetic update operator argument, \n
			i.e. e^{-1j*hbar_*K*dt} is the op

		:return: solver object
		"""
		super().__init__(simName = simName, N = N, np = np, data_drops = data_drops,
		padded = padded, cf = cf, L = L, dx = dx, nf = nf, D = D, C = C, Tf = Tf,
		r = r, v = v, dt = dt, mp = mp)

		### simulation parameters
		self.oldUpdateRule = True
		self.rollMaxDens = False
		self.freezeDensity = False
		self.v_max_artificial = 0 # largest physical velocity
		self.psi_self_grav = True

		### superradiance
		self.superradiance = False
		self.phi_gamma = None # superradiant growth term

		### data params
		self.savePsi = True

		### physics parameters
		self.m22 = m22 # float, particle mass [1e-22 eV/c^2]
		self.hbar_ = hbar_ # float, hbar / m [kpc^2 / Myr] 
		self.V_ext_mesh_func = types.FunctionType
									# func, gives the external potential at self.R
									# due to the enclosed mass

		### dynamical paramters
		self.psi = psi # array-like, [nf,N^D] spatial field
		self.K = K # array-like, [N^3] kinetic update operator argument, kx x kx x kx
		self.R = [] # array-like, [N^3] radial position each cell is defined at

		### diagnostic params
		self.rho_alias = 0.


	def SetParams(self, simName = None, N = None, np = None, mp = None, data_drops = None, padded = None,
		cf = None, L = None, dx = None, nf = None, m22 = [], hbar_ = [], 
		D = None, C = None, Tf = None, r = [], v = [], dt = None, gpu = None, 
		psi = [], K = []):
		"""
		sets parameters

		:simName: string, simulation data directory name
		:N: int, sim resolution
		:np: int, number of corpusular particles
		:mp: float, mass of corpuscular particles
		:data_drops: int, number of data outputs
		:padded: bool, pad the density
		:cf: float, courant factor 
		:L: float, box length
		:dx: float, pixel size
		:nf: int, number of fields
		:m22: array-like, [nf] dark matter mass [1e-22 eV/C^2]
		:hbar_: array-like, [nf] hbar/m
		:D: int, number of spatial dimensions
		:C: float, Poisson's constant
		:Tf: float, final sim time
		:r: array-like, [np, D] particle positions
		:v: array-like, [np, D] particle velocities
		:dt: float, initial timestep
		:gpu: bool, run on the gpu
		:psi: array-like, [nf,N^D] spatial field
		:K: array-like, [N^3] kinetic update operator argument, \n
			i.e. e^{-1j*hbar_*K*dt} is the op
		"""
		super().SetParams(simName = simName, N = N, np = np, data_drops = data_drops,
		padded = padded, cf = cf, L = L, dx = dx, nf = nf, mp = mp,
		D = D, C = C, Tf = Tf, r = r, v = v, dt = dt, gpu = gpu)

		if not(N is None):
			self.set_N(N)
		if not(L is None):
			self.set_L(L)
		if len(hbar_) > 0:
			self.set_hbar_(hbar_)
		if len(m22) > 0:
			self.set_m22(m22)
		if len(hbar_) > 0:
			self.set_hbar_(hbar_)
		if len(psi) > 0:
			self.set_psi(psi)
		if len(K) > 0:
			self.set_K(K)

		if not(N is None):
			self.set_N_perif()
		if not(L is None):
			self.set_L_perif()
		if len(psi) > 0:
			self.set_psi_perif()

	def set_N(self, N):
		"""
		set the grid resolution

		:N: int, grid resolution
		"""
		super().set_N(N)

	def set_N_perif(self):
		"""
		private function, initializes attributes that depend on N
		"""
		super().set_N_perif()
		self.set_K()

	def set_L(self, L):
		"""
		set the box size

		:L: float, box size
		"""
		super().set_L(L)

	def set_L_perif(self):
		"""
		private function, initializes attributes that depend on L
		"""
		super().set_L_perif()
		self.set_K()

	def set_psi(self, psi):
		"""
		set the field

		:psi: array-like, [nf,N^D] spatial field
		"""
		self.psi = psi
		self.set_psi_perif()

	def set_psi_perif(self):
		"""
		private function, sets attributes that depend on psi
		"""
		self.D = len(self.psi.shape) - 1
		self.nf = len(self.psi)


	def set_K(self, K = []):
		"""
		set the spectral gird

		:K: array-like, [N^3] kinetic update operator argument, \n
			i.e. e^{-1j*hbar_*K*dt} is the op \n
			default: use N,L, and D to figure K out
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if len(K) > 0:
			self.K = K 
		else:
			if self.N != None and self.L != None and self.D != None:
				dx = self.L/self.N
				kx = 2*np.pi*np.fft.fftfreq(self.N,d = dx)
				ones = np.ones(self.N)

				self.K = self.get_K(self.N, self.L)


	def set_m22(self, m22):
		"""
		sets the mass of the field [1e-22 eV/C^2]

		:m22: array-like, [nf] dark matter mass [1e-22 eV/C^2]
		"""
		self.m22 = m22
		self.set_m22_perif()

	def set_hbar_(self, hbar_):
		"""
		sets hbar_ of the field 

		:hbar_: array-like, [nf] hbar/m
		"""		
		self.hbar_ = hbar_ 

	def set_m22_perif(self):
		"""
		private function, set attributes that depend on m22
		"""
		self.hbar_ = .01959 / self.m22
		self.nf = len(self.m22)

	def set_hbar_perif(self):
		"""
		private function, set attributes that depend on m22
		"""
		self.m22 = .01959 / self.hbar_
		self.nf = len(self.hbar_)

	def set_nf_perif(self):
		"""
		private function, initializes attributes that depend on nf
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if len(self.m22) == 0:
			self.m22 = np.zeros(self.nf)
		if len(self.hbar_) == 0:
			self.hbar_ = np.zeros(self.nf)


	def set_psi_from_file(self, fileName):
		"""
		loads a file and sets it to be the field

		:fileName: ICs file name
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		self.set_psi(np.load("ICs/" + fileName))


	def InitializeFiles(self):
		"""
		makes the data directory, outputs the toml file and initial data drop
		"""
		if self.simName == None:
			raise Exception("simName has not been set.\n"+\
				"set simName before initializing files.")

		if not(os.path.isdir(f"Data/{self.simName}")):
			os.mkdir(f"Data/{self.simName}")
		self.OutputToml() # toml file
		self.OutputICs()


	def OutputICs(self):
		"""
		outputs the initial conditions
		"""
		if not(os.path.isdir(f"Data/{self.simName}/r")) and self.np != None and self.np > 0:
			os.mkdir(f"Data/{self.simName}/r")
		if not(os.path.isdir(f"Data/{self.simName}/v")) and self.np != None and self.np > 0:
			os.mkdir(f"Data/{self.simName}/v")
		if not(os.path.isdir(f"Data/{self.simName}/psi")):
			os.mkdir(f"Data/{self.simName}/psi")
		self.DataDrop(0)



	def OutputToml(self):
		"""
		outputs toml with simulation parameters
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		mp = self.mp 
		if (isinstance(mp, float) or isinstance(mp,int)) and not(mp is None):
			mp = np.ones(self.np)*self.mp
		if not(mp is None):
			mp = np.mean(mp)

		padded_ = "true" if self.padded else "false"
		# repr() each element so floats always keep a leading/trailing digit
		# (e.g. 5.0 not 5.), which the strict `toml` parser requires.
		m22_ = "[" + ", ".join(repr(float(x)) for x in np.atleast_1d(self.m22)) + "]"
		# TOML has no null; write nan when there are no particles so the value
		# stays a valid float instead of the invalid literal `None`.
		mp_ = repr(float(mp)) if mp is not None else "nan"
		text = f'''
		# all units in kpc, Msolar, Myr
		[physics]
		Tfinal                      = {self.Tf + self.T_initial} # float, final sim time
		L                           = {self.L} # float, box length
		C 							= {self.C} # float, poisson's constant
		D 							= {self.D} # int, number of spatial dimensions
		m22 						= {m22_} # hbar / m_field

		[simulation]
		N                           = {self.N} # int, grid size
		n_particles                 = {self.np} # int, number of simulation particles
		mp_mean		 				= {mp_} # float, mass of simulation particles
		drops                       = {self.data_drops+ self.initial_drop} # int, number of data drops
		padded 						= {padded_} # bool, pad density with 0s
		c_f                         = {self.cf} # float, timestep courant factor
		'''

		f = open(f"Data/{self.simName}/meta.toml", "w")
		f.write(text)
		f.close()

		if len(self.extras) > 0:
			extras = {}
			extras['extras'] = self.extras
			su.AddLines2Toml(extras, self.GetTomlString())


	def DataDrop(self, i):
		"""
		output the current status of the dynamical variables
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		super().DataDrop(i)
		if self.savePsi or (i == 0 or i==1 or i == self.data_drops // 2 or i == self.data_drops // 3 or i == self.data_drops):
			np.save("Data/" + self.simName + f"/psi/drop{i+ self.initial_drop}.npy", self.psi)


	def M_encl_to_phi(self):
		"""
		compute the potential from enclosed mass
		"""
		return self.V_ext_mesh_func(self.R)



	def compute_phi(self, include_particles = True, include_external = True):
		"""
		compute the potential

		:include_particles: bool, include particle density in phi calc
		:include_external: bool, include effect or external forces
	
		:return: array-like, [N^D]
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		rval = np.sum(np.abs(self.psi)**2, axis = 0)

		particles_have_mass = False
		mp = self.mp
		if (isinstance(mp, float) or isinstance(mp,int)):
			particles_have_mass = mp > 0
		elif self.np > 0 and len(mp) > 0:
			particles_have_mass = np.mean(mp) > 0

		if particles_have_mass and len(self.r) > 0 \
		 and include_particles:
			rval += self.ComputeParticleDensity()

		if self.padded:
			rval = self.pad_density(rval)

		rval = self.GetFFt(rval, NoFieldDimension=True)

		K = self.K
		if self.padded:
			K = self.get_K(self.N*2, self.L*2)
		rval = -1*self.C*rval / K
		if self.D == 3:
			rval[0,0,0] = 0.0
		elif self.D ==2:
			rval[0,0] = 0.
		elif self.D == 1:
			rval[0] = 0

		rval = self.GetFFt(rval, Forward = False, NoFieldDimension=True)
		rval = rval.real

		if self.padded:
			if self.D == 3:
				rval = rval[:self.N,:self.N,:self.N]
			elif self.D == 2:
				rval = rval[:self.N,:self.N]
			elif self.D == 1:
				rval = rval[:self.N]

		if ((len(self.M_encl) > 0 and len(self.r_ext) > 0) or self.M_encl_func_on)\
			and include_external:
			rval += self.M_encl_to_phi()
			
		if len(self.V_ext) > 0:
			if len(rval) == 0:
				rval = np.zeros( np.shape(self.V_ext) )
			rval += self.V_ext

		return rval.real


	def get_dt(self, T_remaining, Vmax = None):
		"""
		calculate the timestep

		:T_remaining: float, the time remaining until the next data drop, \n
					default: do not consider this condition
		:Vmax: float, max value of the potential, default: calculate Vmax
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		dt_base = super().get_dt(T_remaining)
		
		if self.freezeDensity:
			return dt_base

		k_max = np.sqrt(self.D) * np.pi / self.dx
		if (self.v_max_artificial > 0):
			k_max_artificial = self.v_max_artificial / self.hbar_
			if (k_max_artificial < k_max):
				k_max = k_max_artificial
		delta_k = 2 * k_max / self.N

		dt_kinetic = self.cf * 2. * np.pi / k_max / delta_k / np.max(self.hbar_)
		dt_kinetic_old = self.cf * 4 *np.pi / k_max**2 / np.max(self.hbar_)
		dt_kinetic = dt_kinetic if not(self.oldUpdateRule) else dt_kinetic_old

		dt_potential = T_remaining		
		if self.psi_self_grav:
			if Vmax == None:
				phi = self.compute_phi()
				phi -= np.mean(phi)
				Vmax = np.max(np.abs(phi)) if self.oldUpdateRule else self.GetMaxPhiGrad(phi)
			dt_potential = self.cf * 2 * np.pi * np.min(self.hbar_) / Vmax
		# rho = self.ComputeParticleDensity()

		dt_array = np.zeros(3)
		dt_array[0] = dt_base
		dt_array[1] = dt_kinetic
		dt_array[2] = dt_potential
		# print(dt_array)
		# assert(0)

		dt_rval = np.min(dt_array)
		if (dt_rval < 0):
			print("negative time!!!")
			print(dt_array)
			assert(0)

		return dt_rval


	def Drift(self, dt):
		"""
		update the dynamic variable positions

		:dt: float, timestep
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if not(self.freezeDensity):
			k2max = (np.pi*self.N/self.L)**2

			self.psi = self.GetFFt(self.psi)

			if self.D == 3:
				self.psi *= np.exp(-1j*dt*\
		            np.einsum("i,jkl->ijkl",self.hbar_, self.K)/(2.))
			elif self.D == 2:
				self.psi *= np.exp(-1j*dt*\
		            np.einsum("i,jk->ijk",self.hbar_, self.K)/(2.))
			elif self.D == 1:
				self.psi *= np.exp(-1j*dt*\
		            np.einsum("i,j->ij",self.hbar_, self.K)/(2.))


			if self.nf == 1:
				if self.D == 3:
					rhoOverThresh = np.sum(np.abs(self.psi[self.K[np.newaxis,:,:,:] > (k2max*.9)])**2)
				elif self.D == 2:
					rhoOverThresh = np.sum(np.abs(self.psi[self.K[np.newaxis,:,:] > (k2max*.9)])**2)
				elif self.D == 1:
					rhoOverThresh = np.sum(np.abs(self.psi[self.K[np.newaxis,:] > (k2max*.9)])**2)
			else:
				rhoOverThresh = 0
			
			rhoToT = np.sum(np.abs(self.psi)**2)
			self.rho_alias = rhoOverThresh / rhoToT

			self.psi = self.GetFFt(self.psi, Forward=False)

		super().Drift(dt)


	def RollSim(self, r_mean = None):

		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		# rho = np.sum(np.abs(self.psi)**2, axis = 0)

		# ind_max = np.argmax(rho)
		x = self.dx*(.5+np.arange(-1*self.N//2, self.N//2))

		### find the mean particle position
		### find the index corresponding to that position
		### roll the sim by that index
		if (self.np is None) or self.np == 0:
			return

		if r_mean is None:
			r_mean = np.mean(self.r, axis = 0)
			r_mean += self.L / 2.

		inds = r_mean / self.dx
		i = int(inds[0])
		x_ = x[i]
		self.r[:,0] -= x_
		self.psi = np.roll(self.psi, self.N//2 - i, axis = 1)
		
		if self.D > 1:
			j = int(inds[1])
			y_ = x[j]
			self.r[:,1] -= y_
			self.psi = np.roll(self.psi, self.N//2 - j, axis = 2)
		if self.D > 2:
			k = int(inds[2])
			z_ = x[k]
			self.r[:,2] -= z_
			self.psi = np.roll(self.psi, self.N//2 - k, axis = 3)		



	def Kick(self, dt):
		"""
		update the dynamic variable momenta

		:dt: float, timestep
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		if self.psi_self_grav or not(self.explicit_particle_forces):
			phi = self.compute_phi(include_particles = True, include_external= True)

		if not(self.freezeDensity) and self.psi_self_grav:
			if self.D == 3:
				self.psi *= np.exp(-1j*dt*\
	            	np.einsum("i,jkl->ijkl",1./self.hbar_, phi))
			elif self.D == 2:
				self.psi *= np.exp(-1j*dt*\
	            	np.einsum("i,jk->ijk",1./self.hbar_, phi))
			elif self.D == 1:
				self.psi *= np.exp(-1j*dt*\
	            	np.einsum("i,j->ij",1./self.hbar_, phi))


		if not(self.np is None) and self.np > 0:
			# external potentials are included in super().ComputeAcc
			# so we don't need to double count them here
			if self.explicit_particle_forces:
				phi = self.compute_phi(include_particles = False, include_external=False)
			elif (len(self.M_encl) > 0 and len(self.r_ext) > 0) or self.M_encl_func_on:
				phi = self.compute_phi(include_external=False)
			acc = self.ComputeAcc(phi)
			self.v += acc*dt

		Vmax = np.max(np.abs(phi - np.mean(phi)))

		if not(self.oldUpdateRule):
			Vmax = self.GetMaxPhiGrad(phi)

		return Vmax


	def GetMaxPhiGrad(self,phi):
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		Vmax = np.abs(mu.gradient_1D(phi, self.dx, axis_ = 0, gpu = self.gpu)[2:self.N-2,2:self.N-2,2:self.N-2])**2
		Vmax += np.abs(mu.gradient_1D(phi, self.dx, axis_ = 1, gpu = self.gpu)[2:self.N-2,2:self.N-2,2:self.N-2])**2
		Vmax += np.abs(mu.gradient_1D(phi, self.dx, axis_ = 2, gpu = self.gpu)[2:self.N-2,2:self.N-2,2:self.N-2])**2
		Vmax = np.sqrt(np.max(Vmax))*self.dx

		return Vmax


	def Update(self, dt):
		"""
		updates the dynamic variables using a drift-kick-drift scheme

		:dt: float, the timestep
		"""
		self.Drift(dt/2.)
		Vmax = self.Kick(dt)
		self.Drift(dt/2.)

		if (self.superradiance):
			self.PerformSuperradiance(dt)

		return Vmax


	def PerformSuperradiance(self, dt):
		self.psi += self.phi_gamma * dt

	def PrintDiagnostics(self, T, time0):
		"""
		prints diagnostic information

		:T: float, sim time completed
		:time0: float, real time the sim started
		"""
		str_ = self.BaseDiagnostics(T,time0)
		str_ += " %.2f alias fraction"%(self.rho_alias)
		su.repeat_print(str_)


	def RunSim(self):
		"""
		run the simulation
		"""
		self.checkPotentialConsistent()
		self.InitializeFiles()
		print("\nrunning simulation " + self.simName + "..." )
		time0 = time.time()
		tNext = float(self.Tf)/self.data_drops
		drop = 1
		T = 0.
		Vmax = None

		if (self.make_periodic):
			self.MakePeriodic()

		while(T < self.Tf):
			T_remaining = tNext - T
			dt = self.get_dt(T_remaining, Vmax=Vmax)
			Vmax = self.Update(dt)
			T += dt 
			self.PrintDiagnostics(T, time0)
			if T_remaining <= 0:
				# su.PrintTimeUpdate(drop,self.data_drops,time0)
				self.DataDrop(drop)
				drop += 1
				tNext = float(self.Tf*drop)/self.data_drops
				if self.rollMaxDens and drop % 4 == 0:
					self.RollSim()

		if (drop == self.data_drops):
			self.DataDrop(drop)

		su.PrintCompletedTime(time0, "simulation")


	# TODO: 
	# - refactor getDensity
	# - refactor module
	# - add this function to super and then just call with my density
	def ShowDensity(self, name = 'density'):
		"""
		plots density projection and slice

		:name: string, name of saved plot, default: 'density'

		:returns: fig-obj, the fig object with the plot
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		rval = np.sum(np.abs(self.psi)**2, axis = 0)

		particles_have_mass = False
		mp = self.mp
		if (isinstance(mp, float) or isinstance(mp,int)):
			particles_have_mass = mp > 0
		elif mp != None:
			particles_have_mass = np.mean(mp) > 0

		if particles_have_mass and len(self.r) > 0:
			rval += self.ComputeParticleDensity()

		L = self.L
		N = self.N
		dx = self.L/self.N

		if self.D == 3:
			rho_slice = rval[N//2,:,:]
			Sigma = np_.sum(rval, axis = 2)
			rho_slice, Sigma = su.gpu2cpu(rho_slice, Sigma)

			fo = pu.FigObj(2)
			fo.AddDens2d(np_.array([L/-2.,L/2]),np_.log(Sigma) )
			fo.SetTitle("density projection")
			fo.AddDens2d([L/-2.,L/2],np_.log(rho_slice))
			fo.SetTitle("density slice")
			fo.save(name)
			# fo.show()

			return fo

	def GetFieldDensity(self):
		'''
		returns field density
		'''
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		return np.sum(np.abs(self.psi)**2, axis = 0)
