# pylint: disable=C,W
import numpy as np_
import os
import time
import mathUtils as mu
import sysUtils as su
import gridUtils as gu
import plotUtils as pu
import types
import astroUtils as au
CUPY_IMPORTED = True
import warnings as warn 
try:
	import cupy as cp 
except ImportError:
	CUPY_IMPORTED = False

class Solver():

	def __init__(self, simName = None, N = None, np = None, mp = None, data_drops = None,
		padded = False, cf = None, L = None, dx = None, nf = None,
		D = None, C = None, Tf = None, r = [], v = [], dt = None, gpu = False):
		"""
		initialize solver object

		:simName: string, simulation data directory name
		:N: int, sim resolution
		:np: int, number of corpusular particles
		:mp: float (or array-like), mass of corpuscular particles
		:data_drops: int, number of data outputs
		:padded: bool, pad the density
		:cf: float, courant factor 
		:L: float, box length
		:dx: float, pixel size
		:nf: int, number of fields
		:D: int, number of spatial dimensions
		:C: float, Poisson's constant
		:Tf: float, final sim time
		:r: array-like, [np, D] particle positions
		:v: array-like, [np, D] particle velocities
		:dt: float, initial timestep
		:gpu: bool, run on the gpu

		:return: solver object
		"""

		### simulation paramters
		self.simName = simName
		self.N = N # int, sim resolution
		self.np = np # int, number of corpuscular particles
		self.mp = mp # float, mass of corpuscular particles
		self.data_drops = data_drops # int, number of data drops
		self.initial_drop = 0
		self.T_initial = 0.
		self.padded = padded # bool, pad density with 0s
		self.cf = cf # float, courant factor
		self.gpu = gpu # bool, run on gpu
		self.explicit_particle_forces = False # use np^2 algorithm for particle forces
		self.eps = 0. # softening length
		self.dt_max = 1.
		self.make_periodic = True # should the walls of the box identify
		self.mod_positions = False # when performing explicit force calc should particles be centered in box?
		self.v_max_part_artificial = -1

		### physics parameter
		self.L = L # float, box length
		self.dx = dx # float, pixel size
		self.nf = nf # int, number of fields
		self.D = D # int, number of spatial dimensions
		self.C = C # float, poisson's constant
		self.Tf = Tf # float, final time

		### external potentials 
		### WARNING: you should only pick one way to define these 
		### e.g. 
		self.V_ext = [] # array-like, [N^3] external potential defined on grid
		self.M_encl = [] # array-like, 1D spherically symmetric enclosed Mass 
		self.M_encl_func = types.FunctionType
		self.M_encl_func_on = False
		self.r_ext = [] # array-like, 1D radii that M_encl are defined at

		### dynamical paramters
		self.r = r # array-like, [np, D] positions of particles
		self.v = v # array-like, [np, D] velocities of particles
		self.dt = dt # float, timestep

		### diagnostics
		self.times = []
		self.percents = []
		self.counter = 0
		self.timesMax = 100

		self.extras = {} # dictionary to add to toml

	def SetParams(self, simName = None, N = None, np = None, mp = None, data_drops = None,
		padded = None, cf = None, L = None, dx = None, nf = None,
		D = None, C = None, Tf = None, r = [], v = [], dt = None, gpu = None):
		"""
		sets parameters

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
		:D: int, number of spatial dimensions
		:C: float, Poisson's constant
		:Tf: float, final sim time
		:r: array-like, [np, D] particle positions
		:v: array-like, [np, D] particle velocities
		:dt: float, initial timestep
		:gpu: bool, run on the gpu
		"""
		if gpu != None:
			self.set_gpu(gpu)
		if simName != None:
			self.set_simName(simName)
		if N != None:
			self.set_N(N)
		if np != None:
			self.set_np(np)
		if not(mp is None):
			self.set_mp(mp)
		if cf != None:
			self.set_cf(cf)
		if data_drops != None:
			self.set_data_drops(data_drops)
		if padded != None:
			self.set_padded(padded)
		if L != None:
			self.set_L(L)
		if dx != None:
			self.set_dx(dx)
		if nf != None:
			self.set_nf(nf)
		if D != None:
			self.set_D(D)
		if C != None:
			self.set_C(C)
		if Tf != None:
			self.set_Tf(Tf)
		if len(r) > 0:
			self.set_r(r)
		if len(v) > 0:
			self.set_v(v)

		if N != None:
			self.set_N_perif()
		if np != None:
			self.set_np_perif()
		if L != None:
			self.set_L_perif()
		if len(r) > 0:
			self.set_r_perif()
		if len(v) > 0:
			self.set_v_perif()


	def set_simName(self, simName):
		"""
		set the simName

		:simName: string
		"""
		self.simName = simName


	def set_N(self, N):
		"""
		set the grid resolution

		:N: int, grid resolution
		"""
		self.N = N
		self.set_N_perif()

	def set_N_perif(self):
		"""
		private function, initializes attributes that depend on N
		"""
		if self.L != None:
			self.dx = self.L / self.N 

	def set_np(self, np):
		"""
		set the number of corpuscular particles

		:np: int, number of particles
		"""
		self.np = np
		self.set_np_perif()

	def set_np_perif(self):
		"""
		private function, initializes attributes that depend on np
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if len(self.r) == 0 and self.D != None:
			self.r = np.zeros((self.np, self.D))
			self.v = np.zeros((self.np, self.D))

	def set_mp(self, mp):
		"""
		set the mass of corpuscular particles

		:mp: float, mass of particles
		"""
		self.mp = mp

	def set_data_drops(self, data_drops):
		"""
		sets the dumber of data drops in the simulation

		:data_drops: int, number of data drops
		"""
		self.data_drops = data_drops


	def set_padded(self, padded):
		"""
		set whether or not to pad the density when calculating the potential

		:padded: bool, add padding to density
		"""
		self.padded = padded


	def set_cf(self, cf):
		"""
		set the courant factor for simulation timesteps

		:cf: float, the courant factor 
		"""
		self.cf = cf


	def set_L(self, L):
		"""
		set the box length

		:L: float, the box length
		"""
		self.L = L
		self.set_L_perif()


	def set_L_perif(self):
		"""
		private function, sets attributes that depend on L
		"""
		if self.N != None:
			self.dx = self.L/self.N

	def set_dx(self,dx):
		"""
		set the grid resolution

		:dx: float, grid resolution
		"""
		self.dx = dx


	def set_nf(self, nf):
		"""
		set the number of fields

		:nf: int, number of fields
		"""
		self.nf = nf 

	def set_D(self, D):
		"""
		set the number of spatial dimensions

		:D: int
		"""
		self.D = D

	def set_C(self, C):
		"""
		set Poisson's constant

		:C: float
		"""
		self.C = C 

	def set_Tf(self, Tf):
		"""
		set final sim time

		:Tf: float
		"""
		self.Tf = Tf

	def set_r(self, r):
		"""
		set particle positions

		:r: array-like, [np, D] particle positions
		"""
		self.r = r
		self.set_r_perif()

	def set_r_perif(self):
		"""
		private function, initializes values that depend on r
		"""
		self.np = len(self.r)
		self.D = self.r.shape[1]

	def set_v_perif(self):
		"""
		private function, initializes values that depend on v
		"""
		self.np = len(self.v)
		self.D = self.v.shape[1]


	def set_v(self, v):
		"""
		set particle velocities

		:v: array-like, [np, D] particle velocities
		"""
		self.v = v
		self.set_v_perif()

	def set_dt(self, dt):
		"""
		set the initial timestep

		:dt: float, initial timestep
		"""
		self.dt = dt

	def set_gpu(self, gpu):
		"""
		set gpu boolean

		:gpu: bool, run on the gpu
		"""
		self.gpu = gpu

	def ParticlesHaveMass(self):
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		particles_have_mass = False
		mp = self.mp
		if (isinstance(mp, float) or isinstance(mp,int)):
			particles_have_mass = np.abs(mp) > 0
		elif self.np > 0 and len(mp) > 0:
			particles_have_mass = np.abs(np.mean(mp)) > 0 or np.abs(mp[0]) > 0

		return particles_have_mass

	def InitializeFiles(self):
		"""
		makes the data directory and outputs the toml file
		"""
		if self.simName == None:
			raise Exception("simName has not been set.\n"+\
				"set simName before initializing files.")

		if not(os.path.isdir(f"Data/{self.simName}")):
			os.mkdir(f"Data/{self.simName}")
		self.OutputToml()
		self.OutputICs()

	def OutputToml(self):
		"""
		outputs toml with simulation parameters
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		mp = self.mp 
		if (isinstance(mp, float) or isinstance(mp,int)):
			mp = np.ones(self.np)*self.mp
		mp = np.mean(mp)

		text = f'''
		# all units in kpc, Msolar, Myr
		[physics]
		Tfinal                      = {self.Tf + self.T_initial} # float, final sim time
		L                           = {self.L} # float, box length
		C 							= {self.C} # float, poisson's constant
		D 							= {self.D} # int, number of spatial dimensions

		[simulation]
		N                           = {self.N} # int, grid size
		n_particles                 = {self.np} # int, number of simulation particles
		mp_mean		                = {mp} # int, number of simulation particles
		drops                       = {self.data_drops + self.initial_drop} # int, number of data drops
		padded 						= {str(self.padded).lower()} # bool, pad density with 0s
		c_f                         = {self.cf} # float, timestep courant factor
		'''

		f = open(f"Data/{self.simName}/meta.toml", "w")
		f.write(text)
		f.close()

		if len(self.extras) > 0:
			extras = {}
			extras['extras'] = self.extras
			su.AddLines2Toml(extras, self.GetTomlString())


	def GetTomlString(self):
		"""
		returns the name of the tomlFile
		"""
		return f"Data/{self.simName}/meta.toml"



	def OutputICs(self):
		"""
		outputs the initial conditions
		"""
		if not(os.path.isdir(f"Data/{self.simName}/r")) and self.np != None and self.np > 0:
			os.mkdir(f"Data/{self.simName}/r")
		if not(os.path.isdir(f"Data/{self.simName}/v")) and self.np != None and self.np > 0:
			os.mkdir(f"Data/{self.simName}/v")
		self.DataDrop(0)


	def DataDrop(self, i):
		"""
		output the current status of the dynamical variables
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		if self.np != None and self.np > 0:
			np.save("Data/" + self.simName + f"/r/drop{i + self.initial_drop}.npy", self.r)
			np.save("Data/" + self.simName + f"/v/drop{i + self.initial_drop}.npy", self.v)


	def get_dt(self, T_remaining):
		"""
		calculate the timestep

		:T_remaining: float, the time remaining until the next data drop
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if self.np == None or self.np == 0:
			return T_remaining

		dt_all = np.zeros(3)

		max_v = np.max(np.abs(self.v))
		if self.v_max_part_artificial > 0 and self.v_max_part_artificial < max_v:
			max_v = self.v_max_part_artificial

		dt_all[0] = self.cf * self.dx / max_v
		dt_all[1] = T_remaining
		dt_all[2] = self.dt_max

		dt_rval = np.min(dt_all)
		if (dt_rval < 0):
			print("negative time!!!")
			print(dt_all)

		return np.min(dt_all)
		
	def GetFFt(self, psi, Forward = True, NoFieldDimension = False):
		"""
		calculate the fft of psi

		:psi: array-like, [nf, N^D], the fields
		:Forward: bool, forward or backward fft, default: True
		:NoFieldDimension: bool, if True then psi has dim[N^D]

		:return: array-like, fft of psi on all axes except 0th
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if self.D == None:
			self.D = len(psi.shape) - 1

		a = NoFieldDimension

		if Forward:
			if self.D == 1:
				return np.fft.fft(psi, axis = 1 - a)
			elif self.D == 2:
				return np.fft.fft2(psi, axes = (1-a,2-a))
			elif self.D == 3:
				return np.fft.fftn(psi, axes = (1-a,2-a,3-a))
		else:
			if self.D == 1:
				return np.fft.ifft(psi, axis = 1-a)
			elif self.D == 2:
				return np.fft.ifft2(psi, axes = (1-a,2-a))
			elif self.D == 3:
				return np.fft.ifftn(psi, axes = (1-a,2-a,3-a))


	def pad_density(self, rho):
		"""
		puts rho in the bottom corner of zeros
		
		:rho: array-like, density

		:returns: array-like, zeros with bottom corner equal to rho
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		if self.D == None:
			self.D = len(rho.shape)

		if self.D == 1:
			zeros = np.zeros(self.N*2)
			zeros[:self.N] = rho
		if self.D == 2:
			zeros = np.zeros((self.N*2, self.N*2))
			zeros[:self.N, :self.N] = rho
		if self.D == 3:
			zeros = np.zeros((self.N*2, self.N*2, self.N*2))
			zeros[0:self.N, 0:self.N, 0:self.N] = rho

		return zeros

	def get_K(self, N, L):
		"""
		calculate the spectral grid

		:N: int, the grid resolution
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		dx = L/N
		kx = 2*np.pi*np.fft.fftfreq(N, d = dx)
		ones = np.ones(N)

		if self.D == 1:
			return kx**2
		elif self.D == 2:
			K = np.einsum("i,j->ij", kx**2, ones)
			K += np.einsum("i,j->ij", ones, kx**2)
			return K
		elif self.D == 3:
			K = np.einsum("i,j,k->ijk", kx**2, ones, ones)
			K += np.einsum("i,j,k->ijk", ones, kx**2, ones)
			K += np.einsum("i,j,k->ijk", ones, ones, kx**2)
			return K

	def compute_phi(self, padded = None):
		"""
		compute the potential

		:return: array-like, [N^D]
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if padded == None:
			padded = self.padded

		rval = []

		if not(self.explicit_particle_forces):
			# calculate the particle self grav with a mesh
			rval = self.ComputeParticleDensity()

			if padded:
				rval = self.pad_density(rval)

			rval = self.GetFFt(rval,NoFieldDimension=True)

			K = None 
			if padded:
				K = self.get_K(self.N*2, self.L*2)
			else: 
				K = self.get_K(self.N, self.L)
			rval *= -1*self.C / K
			if self.D == 3:
				rval[0,0,0] = 0.0
			elif self.D ==2:
				rval[0,0] = 0.
			elif self.D == 1:
				rval[0] = 0

			rval = self.GetFFt(rval, Forward = False,NoFieldDimension=True)

			if self.padded:
				if self.D == 3:
					rval = rval[:self.N,:self.N,:self.N]
				elif self.D == 2:
					rval = rval[:self.N,:self.N]
				elif self.D == 1:
					rval = rval[:self.N]

		if len(self.V_ext) > 0:
			if len(rval) == 0:
				rval = np.zeros( np.shape(self.V_ext) )
			rval += self.V_ext

		return np.real(rval)

	def MakePeriodic(self):
		if len(self.r) > 0:
			if self.make_periodic:
				self.r += self.L/2.
				self.r %= self.L
				self.r -= self.L/2.


	def Drift(self, dt):
		"""
		update the dynamic variable positions

		:dt: float, timestep
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if len(self.r) > 0 and len(self.v) > 0:
			self.r += self.v * dt
			if self.make_periodic:
				self.MakePeriodic()
				# self.r += self.L/2.
				# self.r %= self.L
				# self.r -= self.L/2.
				# self.r[self.r < -1*self.L/2.] += self.L
				# self.r[self.r > self.L/2.] -= self.L

	def Kick(self, dt):
		"""
		update the dynamic variable momenta

		:dt: float, timestep
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		phi = self.compute_phi()

		acc = self.ComputeAcc(phi)
		self.v += acc*dt

		if len(phi) > 0:
			return np.max(phi)
		else:
			return np.max(acc)


	def Update(self, dt):
		"""
		updates the dynamic variables using a drift-kick-drift scheme

		:dt: float, the timestep
		"""
		self.Drift(dt/2.)
		Vmax = self.Kick(dt)
		self.Drift(dt/2.)
		return Vmax


	def checkPotentialConsistent(self):
		"""
		checks to make sure I am not using an inconsistent external potential definitions
		"""
		methods = 0

		# if len(self.V_ext):
		# 	# trying to define external potential on grid
		# 	methods += 1
		if len(self.M_encl) and len(self.r_ext):
			# trying to define external potential using explicit enclosed mass
			methods += 1
		if self.M_encl_func_on:
			methods += 1

		if methods > 1:
			raise Exception("external potential overdefined.\n" +\
				"pick one method.")


	def RunSim(self):
		"""
		run the simulation
		"""
		self.InitializeFiles()
		print("\nrunning simulation " + self.simName + "..." )
		time0 = time.time()
		tNext = float(self.Tf)/self.data_drops
		drop = 1
		T = 0.

		if (self.make_periodic):
			self.MakePeriodic()

		while(T < self.Tf):
			T_remaining = tNext - T
			dt = self.get_dt(T_remaining)
			Vmax = self.Update(dt)
			T += dt 
			self.PrintDiagnostics(T, time0)
			if T_remaining <= 0:
				# su.PrintTimeUpdate(drop,self.data_drops,time0)
				self.DataDrop(drop)
				drop += 1
				tNext = float(self.Tf*drop)/self.data_drops

		if (drop == self.data_drops):
			self.DataDrop(drop)

		su.PrintCompletedTime(time0, "simulation")


	def BaseDiagnostics(self, T, time0):
		"""
		prints diagnostic information

		:T: float, sim time completed
		:time0: float, real time the sim started

		:return: string, diagnostic info
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		T_remaining = self.Tf - T
		elapsedTimeTotal = time.time() - time0
		portionDone = T / self.Tf
		portionRemaining = 1 - portionDone

		if self.counter == 0:
			self.times = np.zeros(self.timesMax)
			self.percents = np.zeros(self.timesMax)

		ind_ = self.counter%self.timesMax
		self.times[ind_] = time.time()
		self.percents[ind_] = portionDone

		elapsedTimeRecently = elapsedTimeTotal
		portionDoneRecently = portionDone

		if self.counter == self.timesMax:
			elapsedTimeRecently = \
				(np.max(self.times) - np.min(self.times)) / (len(self.times) - 1)
			portionDoneRecently = \
				(np.max(self.percents) - np.min(self.percents)) / (len(self.percents) - 1)

		self.counter += 1

		string_done = "%.2f done "%(portionDone) 
		string_done += "in %i hrs, %i mins, %i s."%su.hms(elapsedTimeTotal)

		time_estimate = portionRemaining*elapsedTimeRecently / portionDoneRecently
		string_todo = f" eta: %i hrs, %i mins, %i s."%su.hms(time_estimate)
		return string_done + string_todo


	def PrintDiagnostics(self, T, time0):
		"""
		prints diagnostic information

		:T: float, sim time completed
		:time0: float, real time the sim started
		"""

		su.repeat_print(self.BaseDiagnostics(T,time0))


	def ComputeAcc_ext(self):
		"""
		compute the acceleration due to the radially symmetric external potential

		:returns: array-like, [np, D] accelerations
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		M_encl_r = np.zeros(self.np)

		R = np.abs(self.r)
		if self.D > 1:
			R = mu.Norm_L2(self.r, axes = (1), gpu = self.gpu)

		if self.M_encl_func_on:
			M_encl_r = self.M_encl_func(R)
		else:
			M_encl_r = np.interp(R, self.r_ext, self.M_encl)
			M_encl_r[R < np.min(self.r_ext)] = 0.
			M_encl_r[R > np.max(self.r_ext)] = np.max(self.M_encl)

		if self.D == 1:
			return -1*self.C*M_encl_r*np.sign(self.r) / (4*np.pi)
		elif self.D == 2:
			return -2*self.C*M_encl_r[:,np.newaxis]*self.r / R[:,np.newaxis]**2 / (4*np.pi)
		elif self.D == 3:
			return -1*self.C*M_encl_r[:,np.newaxis]*self.r / R[:,np.newaxis]**3 / (4*np.pi)


	def ComputeAcc(self, phi):
		"""
		compute the acceleration for the corpuscular particles given a potential

		:phi: array-like, the potential

		:return: array-like, [np, D] acceleration
		"""

		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		acc = np.zeros(np.shape(self.r))

		if len(phi) > 0:
			if self.D == 1:
				acc = self.ComputeAcc1D(phi)
			elif self.D == 2:
				acc = self.ComputeAcc2D(phi)
			elif self.D == 3:
				acc = self.ComputeAcc3D(phi)

		if self.explicit_particle_forces and self.ParticlesHaveMass():
			acc += self.ExplicitParticleForces()

		if (len(self.M_encl) > 0 and len(self.r_ext) > 0) or self.M_encl_func_on:
			acc += self.ComputeAcc_ext()
		

		return acc



	def ExplicitParticleForces(self):
		"""
		compute the explicit acceleration between particles

		:return: array-like, [np, D] acceleration
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp 
		
		mp = self.mp
		if (isinstance(mp, float) or isinstance(mp,int)):
			mp = np.ones(self.np)*self.mp

		acc = np.zeros(np.shape(self.r))

		range_ = np.arange(self.np)
		for i in range_:
			r_ = self.r[i]
			mp_ = mp[range_ != i]

			if self.D == 3:
				r_vecs = self.r[range_ != i,:] - r_[np.newaxis,:]
				if self.mod_positions:
				    r_vecs[r_vecs > self.L/2.] -= self.L
				    r_vecs[r_vecs < -1*self.L/2.] += self.L

				R = mu.Norm_L2(r_vecs, axes = (1)) + self.eps 
				acc[i] = self.C*np.sum(mp_[:,np.newaxis]*r_vecs / R[:,np.newaxis]**3, axis = 0) / (4*np.pi)
			
			if self.D == 2:
				r_vecs = self.r[range_ != i,:] - r_[np.newaxis,:]
				if self.mod_positions:
				    r_vecs[r_vecs > self.L/2.] -= self.L
				    r_vecs[r_vecs < -1*self.L/2.] += self.L
				R = mu.Norm_L2(r_vecs, axes = (1))  + self.eps
				acc[i] = 2*self.C*np.sum(mp_[:,np.newaxis]*r_vecs / R[:,np.newaxis]**2, axis = 0) / (4*np.pi)
			
			if self.D == 1:
				r_vecs = self.r[range_ != i] - r_
				if self.mod_positions:
				    r_vecs[r_vecs > self.L/2.] -= self.L
				    r_vecs[r_vecs < -1*self.L/2.] += self.L
				acc[i] = self.C*np.sum(mp_*np.sign(r_vecs)) / (4*np.pi)

		return acc


	def ComputeAcc3D(self, phi):
		"""
		compute the 3D acceleration for the corpuscular particles given a potential

		:phi: array-like, the potential

		:return: array-like, [np, D] acceleration
		"""
		N = self.N
		r = self.r
		dx = self.dx
		L = self.L

		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		acc = np.zeros(r.shape)  
		ijk = np.floor((r + L/2.) / dx).astype(int) # [n_particles, [i,j,k]]\
		ijk %= N

		x = dx*(.5+np.arange(-1*N//2, N//2))

		for i in range(self.D):
			axis_ = i 
			gradPhi = mu.gradient_1D(phi, dx, axis_, gpu = self.gpu, padded= self.padded)

			# i,j,k
			f1 = 1 - np.abs(x[ijk[:,0]] - r[:,0])/dx
			f2 = 1 - np.abs(x[ijk[:,1]] - r[:,1])/dx
			f3 = 1 - np.abs(x[ijk[:,2]] - r[:,2])/dx
			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i+1,j,k
			f1 = 1. - f1 # dx
			ijk[:,0] += 1
			ijk[:,0] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i+1,j+1,k
			f2 = 1. - f2 # tx
			ijk[:,1] += 1
			ijk[:,1] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i,j+1,k
			f1 = 1. - f1
			ijk[:,0] -= 1
			ijk[:,0] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i,j+1,k+1
			f3 = 1. - f3
			ijk[:,2] += 1
			ijk[:,2] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i,j,k+1
			f2 = 1. - f2
			ijk[:,1] -= 1
			ijk[:,1] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i+1,j,k+1
			f1 = 1. - f1
			ijk[:,0] += 1
			ijk[:,0] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			# i+1,j+1,k+1
			f2 = 1. - f2
			ijk[:,1] += 1
			ijk[:,1] %= N

			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

			ijk[:,0] -= 1
			ijk[:,1] -= 1
			ijk[:,2] -= 1

		return -1*acc

	def ComputeAcc2D(self, phi):
		"""
		compute the 2D acceleration for the corpuscular particles given a potential

		:phi: array-like, the potential

		:return: array-like, [np, D] acceleration
		"""
		N = self.N
		r = self.r
		dx = self.dx
		L = self.L

		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		acc = np.zeros(r.shape)  
		ijk = np.floor((r + L/2.) / dx).astype(int) # [n_particles, [i,j,k]]\
		ijk %= N

		x = dx*(.5+np.arange(-1*N//2, N//2))

		for i in range(self.D):
			axis_ = i 
			gradPhi = mu.gradient_1D(phi, dx, axis_, gpu = self.gpu, padded=self.padded)

			# i,j
			f1 = 1 - np.abs(x[ijk[:,0]] - r[:,0])/dx
			f2 = 1 - np.abs(x[ijk[:,1]] - r[:,1])/dx
			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1]]*f1*f2

			# i+1,j
			f1 = 1. - f1 # dx
			ijk[:,0] += 1
			ijk[:,0] %= N
			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1]]*f1*f2

			# i+1,j+1
			f2 = 1. - f2 # tx
			ijk[:,1] += 1
			ijk[:,1] %= N
			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1]]*f1*f2

			# i,j+1
			f1 = 1. - f1
			ijk[:,0] -= 1
			ijk[:,0] %= N
			acc[:,i] += gradPhi[ijk[:,0],ijk[:,1]]*f1*f2

			ijk[:,0] -= 1
			ijk[:,1] -= 1

		return -1*acc


	def ComputeAcc1D(self, phi):
		"""
		compute the 1D acceleration for the corpuscular particles given a potential

		:phi: array-like, the potential

		:return: array-like, [np, D] acceleration
		"""
		N = self.N
		r = self.r
		dx = self.dx
		L = self.L

		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		acc = np.zeros(r.shape)  
		ijk = np.floor((r + L/2.) / dx).astype(int) # [n_particles, [i,j,k]]\
		ijk %= N

		x = dx*(.5+np.arange(-1*N//2, N//2))

		axis_ = 0 
		gradPhi = mu.gradient_1D(phi, dx, axis_, gpu = self.gpu, padded=self.padded)

		# i
		f1 = 1 - np.abs(x[ijk[:]] - r[:])/dx
		acc[:] += gradPhi[ijk[:]]*f1

		# i+1
		f1 = 1. - f1 # dx
		ijk[:] += 1
		ijk[:] %= N
		acc[:] += gradPhi[ijk[:]]*f1

		return -1*acc


	def ComputeParticleDensity(self):
		"""
		compute the density for the corpuscular particles

		:return: array-like, [N^D] density
		"""
		if self.D == 1:
			return self.ComputeParticleDensity1D()
		elif self.D == 2:
			return self.ComputeParticleDensity2D()
		elif self.D == 3:
			return self.ComputeParticleDensity3D()


	def ComputeParticleDensity3D(self):
		"""
		compute the 3D density for the corpuscular particles

		:return: array-like, [D^3] density
		"""
		N = self.N
		r = self.r
		dx = self.dx
		L = self.L
		len_ = N*N*N

		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		mp = self.mp
		if (isinstance(mp, float) or isinstance(mp,int)):
			mp = np.ones(self.np)*self.mp

		rho_r = np.zeros((len_))
		ijk = np.floor((r + L/2.) / dx).astype(int) # [n_particles, [i,j,k]]\
		ijk %= N
		# print(np.min(rho_r), np.max(rho_r))


		x = dx*(.5+np.arange(-1*N//2, N//2))

		# i,j,k
		f1 = 1 - np.abs(x[ijk[:,0]] - r[:,0])/dx
		f2 = 1 - np.abs(x[ijk[:,1]] - r[:,1])/dx
		f3 = 1 - np.abs(x[ijk[:,2]] - r[:,2])/dx
		# f2 = (x[ijk[:,1]] - r[:,1])/dx + 1. # ty
		# f3 = (x[ijk[:,2]] - r[:,2])/dx + 1. # ty
		ind_ = ijk[:,0]*N**2 + ijk[:,1]*N + ijk[:,2]
		rho_ = np.bincount(ind_, weights = f1*f2*f3*mp, minlength = len_)
		# print(np.min(rho_), np.max(rho_))
		rho_r += rho_

		# i+1,j,k
		f1 = 1. - f1 # dx
		ijk[:,0] += 1
		ijk[:,0] %= N
		ind_ = ijk[:,0]*N**2 + ijk[:,1]*N + ijk[:,2]
		rho_ = np.bincount(ind_, weights = f1*f2*f3*mp, minlength = len_)
		# print(np.min(rho_), np.max(rho_))
		rho_r += rho_

		# i+1,j+1,k
		f2 = 1. - f2 # tx
		ijk[:,1] += 1
		ijk[:,1] %= N
		ind_ = ijk[:,0]*N**2 + ijk[:,1]*N + ijk[:,2]
		rho_r += np.bincount(ind_, weights = f1*f2*f3*mp, minlength = len_)

		# i,j+1,k
		f1 = 1. - f1
		ijk[:,0] -= 1
		ijk[:,0] %= N
		ind_ = ijk[:,0]*N**2 + ijk[:,1]*N + ijk[:,2]
		rho_r += np.bincount(ind_, weights = f1*f2*f3*mp, minlength = len_)

		# i,j+1,k+1
		f3 = 1. - f3
		ijk[:,2] += 1
		ijk[:,2] %= N
		ind_ = ijk[:,0]*N**2 + ijk[:,1]*N + ijk[:,2]
		rho_r += np.bincount(ind_, weights = f1*f2*f3*mp, minlength = len_)

		# i,j,k+1
		f2 = 1. - f2
		ijk[:,1] -= 1
		ijk[:,1] %= N
		ind_ = ijk[:,0]*N**2 + ijk[:,1]*N + ijk[:,2]
		rho_r += np.bincount(ind_, weights = f1*f2*f3*mp, minlength = len_)

		# i+1,j,k+1
		f1 = 1. - f1
		ijk[:,0] += 1
		ijk[:,0] %= N
		ind_ = ijk[:,0]*N**2 + ijk[:,1]*N + ijk[:,2]
		rho_r += np.bincount(ind_, weights = f1*f2*f3*mp, minlength = len_)

		# i+1,j+1,k+1
		f2 = 1. - f2
		ijk[:,1] += 1
		ijk[:,1] %= N
		ind_ = ijk[:,0]*N**2 + ijk[:,1]*N + ijk[:,2]
		rho_r += np.bincount(ind_, weights = f1*f2*f3*mp, minlength = len_)

		dV = dx**3 

		return rho_r.reshape(N,N,N)/dV


	def ComputeParticleDensity2D(self):
		"""
		compute the 2D density for the corpuscular particles

		:return: array-like, [D^2] density
		"""
		N = self.N
		r = self.r
		dx = self.dx
		L = self.L
		len_ = N*N

		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		mp = self.mp
		if (isinstance(mp, float) or isinstance(mp,int)):
			mp = np.ones(self.np)*self.mp

		rho_r = np.zeros((len_))
		ijk = np.floor((r + L/2.) / dx).astype(int) # [n_particles, [i,j,k]]\
		ijk %= N

		x = dx*(.5+np.arange(-1*N//2, N//2))

		# i,j
		f1 = 1 - np.abs(x[ijk[:,0]] - r[:,0])/dx
		f2 = 1 - np.abs(x[ijk[:,1]] - r[:,1])/dx
		ind_ = ijk[:,0]*N + ijk[:,1]
		rho_r += np.bincount(ind_, weights = f1*f2*mp, minlength = len_)

		# i+1,j
		f1 = 1. - f1 # dx
		ijk[:,0] += 1
		ijk[:,0] %= N
		ind_ = ijk[:,0]*N + ijk[:,1]
		rho_r += np.bincount(ind_, weights = f1*f2*mp, minlength = len_)

		# i+1,j+1
		f2 = 1. - f2 # tx
		ijk[:,1] += 1
		ijk[:,1] %= N
		ind_ = ijk[:,0]*N + ijk[:,1]
		rho_r += np.bincount(ind_, weights = f1*f2*mp, minlength = len_)

		# i,j+1
		f1 = 1. - f1
		ijk[:,0] -= 1
		ijk[:,0] %= N
		ind_ = ijk[:,0]*N + ijk[:,1]
		rho_r += np.bincount(ind_, weights = f1*f2*mp, minlength = len_)

		dV = dx**2

		return rho_r.reshape(N,N)/dV


	def ComputeParticleDensity1D(self):
		"""
		compute the 1D density for the corpuscular particles

		:return: array-like, [D] density
		"""
		N = self.N
		r = self.r
		dx = self.dx
		L = self.L
		len_ = N

		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		mp = self.mp
		if (isinstance(mp, float) or isinstance(mp,int)):
			mp = np.ones(self.np)*self.mp

		rho_r = np.zeros((len_))
		ijk = np.floor((r + L/2.) / dx).astype(int) # [n_particles, [i,j,k]]\
		ijk %= N

		x = dx*(.5+np.arange(-1*N//2, N//2))

		# i
		f1 = 1 - np.abs(x[ijk[:]] - r[:])/dx
		ind_ = ijk[:]
		rho_r += np.bincount(ind_, weights = f1*mp, minlength = len_)

		# i+1
		f1 = 1. - f1 # dx
		ijk[:] += 1
		ijk[:] %= N
		ind_ = ijk[:]
		rho_r += np.bincount(ind_, weights = f1*mp, minlength = len_)

		dV = dx

		return rho_r/dV


	def showRadialDensity(self, name = 'density_profile'):
		"""
		plots density projection and slice

		:name: string, name of saved plot, default: 'density_profile'

		:returns: fig-obj, the fig object with the plot
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if self.D == 3:
			rho = self.ComputeParticleDensity3D()
			R, _, _ = gu.sphrGrid(self.N,self.L)
			r, rho_radial = au.radialProfile(R, rho)

			fo = pu.FigObj()
			fo.AddLine(r, rho_radial, label = r'solver density')
			fo.SetXLabel(r'$r$')
			fo.SetYLabel(r'$\rho$')
			fo.SetTitle(r'density profile')
			fo.save(name)

			return fo


	def showRadialPotential(self, name = 'potential_profile'):
		"""
		plots density projection and slice

		:name: string, name of saved plot, default: 'potential_profile'

		:returns: fig-obj, the fig object with the plot
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if self.D == 3:
			phi = self.compute_phi()
			R, _, _ = gu.sphrGrid(self.N,self.L)
			r, phi_radial = au.radialProfile(R, phi)

			fo = pu.FigObj()
			fo.AddLine(r, phi_radial, label = r'solver potential')
			fo.SetXLabel(r'$r$')
			fo.SetYLabel(r'$\phi$')
			fo.SetTitle(r'potential profile')
			fo.save(name)

			return fo, phi	

			



