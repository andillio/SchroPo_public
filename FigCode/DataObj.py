# pylint: disable=C,W
import numpy as np
import astroUtils as au
import sysUtils as su
import sys
sys.path.insert(1, '../')
sys.path.insert(2, '../Solvers')
import Solvers.meshSolver as MS
import Solvers.baseSolver as BS
from enum import Enum
import toml


class DataType(Enum):
	corpuscular = 0
	mesh = 1
	eigenvalue = 2


def DataObj(simName, type):
	pass


class CorpDataObj(BS.Solver):

	def __init__(self, simName):
		'''
		creates a corp sim data object

		:simName: string

		:returns: CorpDataObj
		'''
		super().__init__()
		self.simName = simName # name of simulation
		self.extraParams = {} # dictionary containing extra params
		self.meta = {} # toml values 
		# read in toml
		self.dataDir = f"../Data/{simName}/"
		kwargs = self.readToml(f"../Data/{simName}/meta.toml")
		# assign params to object
		self.SetParams(**kwargs)
		


	def readToml(self, fileName):
		'''
		reads toml and loads data

		:fileName: string, fileName of toml

		:returns: dict, data in toml file
		'''
		meta = toml.load(fileName)
		self.meta = meta

		kwargs = {}

		self.LoadPhysParams(meta, kwargs)
		self.LoadSimParams(meta, kwargs)

		return kwargs


	def LoadPhysParams(self, meta, kwargs):
		'''
		loads the physics parameters section of the toml

		:meta: dict, string dict loaded from toml
		:kwargs: dict, the dictionary to be updated
		'''
		meta_ = meta['physics']

		if "Tfinal" in meta_:
			kwargs["Tf"] = float(meta_["Tfinal"])
		if "L" in meta_:
			kwargs["L"] = float(meta_["L"])
		if "C" in meta_:
			kwargs["C"] = float(meta_["C"])
		if "D" in meta_:
			kwargs["D"] = int(meta_["D"])


	def LoadSimParams(self, meta, kwargs):
		'''
		loads the simulation parameters section of the toml

		:meta: dict, string dict loaded from toml
		:kwargs: dict, the dictionary to be updated
		'''
		meta_ = meta["simulation"]

		if "N" in meta_:
			kwargs["N"] = int(meta_["N"])
		if "n_particles" in meta_:
			kwargs["np"] = int(meta_["n_particles"])
		if "drops" in meta_:
			kwargs["data_drops"] = int(meta_["drops"])
		if "padded" in meta_:
			kwargs["padded"] = bool(meta_["padded"])
		if "c_f" in meta_:
			kwargs["cf"] = float(meta_["c_f"])
		if "mp_mean" in meta_:
			kwargs["mp"] = float(meta_["mp_mean"])


	def LoadExtraParams(self, keys, types = []):
		'''
		loads extra parameters

		:keys: array-like, keys of extra variables
		:types: array-like, types for values

		:returns: dict, the dictionary describing the ectra params
		'''
		kwargs = self.meta['extras']
		self.extraParams = su.ConvertStringDict(keys, types, **kwargs)
		return self.extraParams


	def LoadData(self, drop, returnCopy = True, center = True):
		'''
		loads the dynamic variables

		:drop: int, which data drop should be loaded
		:returnCopy: bool, return copy of (r,v)

		:returns: array-like, (r,v) radius and velocity dynamical variables
		'''
		self.v = np.load(f"../Data/" + self.simName + f"/v/drop{drop}.npy")
		self.r = np.load("../Data/" + self.simName + f"/r/drop{drop}.npy")
		if center:
			self.v -= np.mean(self.v,axis = 0)
			self.r -= np.mean(self.r,axis = 0)
		if returnCopy:
			return self.r.copy(), self.v.copy()


class MeshDataObj(MS.Solver):

	def __init__(self, simName):
		'''
		creates a corp sim data object

		:simName: string

		:returns: CorpDataObj
		'''
		super().__init__()
		self.simName = simName # name of simulation
		self.extraParams = {} # dictionary containing extra params
		self.meta = {} # toml values 
		# read in toml
		self.dataDir = f"../Data/{simName}/"
		kwargs = self.readToml(f"../Data/{simName}/meta.toml")
		# assign params to object
		self.SetParams(**kwargs)

	def readToml(self, fileName):
		'''
		reads toml and loads data

		:fileName: string, fileName of toml

		:returns: dict, data in toml file
		'''
		meta = toml.load(fileName)
		self.meta = meta

		kwargs = {}

		self.LoadPhysParams(meta, kwargs)
		self.LoadSimParams(meta, kwargs)

		return kwargs

	def LoadPhysParams(self, meta, kwargs):
		'''
		loads the physics parameters section of the toml

		:meta: dict, string dict loaded from toml
		:kwargs: dict, the dictionary to be updated
		'''
		meta_ = meta['physics']

		if "Tfinal" in meta_:
			kwargs["Tf"] = float(meta_["Tfinal"])
		if "L" in meta_:
			kwargs["L"] = float(meta_["L"])
		if "C" in meta_:
			kwargs["C"] = float(meta_["C"])
		if "D" in meta_:
			kwargs["D"] = int(meta_["D"])
		if "m22" in meta_:
			kwargs["m22"] = np.array(meta_["m22"], dtype=np.float128)

	def LoadSimParams(self, meta, kwargs):
		'''
		loads the simulation parameters section of the toml

		:meta: dict, string dict loaded from toml
		:kwargs: dict, the dictionary to be updated
		'''
		meta_ = meta["simulation"]

		if "N" in meta_:
			kwargs["N"] = int(meta_["N"])
		if "n_particles" in meta_:
			kwargs["np"] = int(meta_["n_particles"])
		if "drops" in meta_:
			kwargs["data_drops"] = int(meta_["drops"])
		if "padded" in meta_:
			kwargs["padded"] = bool(meta_["padded"])
		if "c_f" in meta_:
			kwargs["cf"] = float(meta_["c_f"])
		if "mp_mean" in meta_:
			kwargs["mp"] = float(meta_["mp_mean"])

	def LoadExtraParams(self, keys, types = []):
		'''
		loads extra parameters

		:keys: array-like, keys of extra variables
		:types: array-like, types for values

		:returns: dict, the dictionary describing the ectra params
		'''
		kwargs = self.meta['extras']
		self.extraParams = su.ConvertStringDict(keys, types, **kwargs)
		return self.extraParams


	def LoadData(self, drop, returnCopy = True, center = True):
		'''
		loads the dynamic variables

		:drop: int, which data drop should be loaded
		:returnCopy: bool, return copy of (r,v)

		:returns: array-like, (r,v) radius and velocity dynamical variables
		'''
		if self.np > 0:
			self.v = np.load(f"../Data/" + self.simName + f"/v/drop{drop}.npy")
			self.r = np.load("../Data/" + self.simName + f"/r/drop{drop}.npy")
			if center:
				self.v -= np.mean(self.v,axis = 0)
				self.r -= np.mean(self.r,axis = 0)
		self.psi = np.load("../Data/" + self.simName + f"/psi/drop{drop}.npy")
		if returnCopy:
			return self.r.copy(), self.v.copy(), self.psi.copy()

	def LoadPsi(self, drop):
		'''
		loads the field 

		:drop: int, which data drop should be loaded

		:returns: array-like, psi
		'''
		self.psi = np.load("../Data/" + self.simName + f"/psi/drop{drop}.npy")
		return self.psi


	def LoadCorpData(self, drop, returnCopy = True, center = True):
		'''
		loads the dynamic variables

		:drop: int, which data drop should be loaded
		:returnCopy: bool, return copy of (r,v)

		:returns: array-like, (r,v) radius and velocity dynamical variables
		'''
		if self.np > 0:
			self.v = np.load(f"../Data/" + self.simName + f"/v/drop{drop}.npy")
			self.r = np.load("../Data/" + self.simName + f"/r/drop{drop}.npy")
			if center:
				self.v -= np.mean(self.v,axis = 0)
				self.r -= np.mean(self.r,axis = 0)
		if returnCopy:
			return self.r.copy(), self.v.copy()
