# pylint: disable=C,W
import numpy as np_
try:
	import cupy as np
except ImportError:
	import numpy as np
import time
import astroUtils as au
import gridUtils as gu
from scipy.special import sici
import sysUtils as su
import plotUtils as pu
import sys
sys.path.insert(1, 'Solvers')
import Solvers.meshSolver as MS
import numpy.linalg as npl
import scipy.stats as sp2

simName = "boxTest1a"
N = 256
data_drops = 20
padded = False
cf = .1 
nf = 1
C = au.G*4*np.pi
Tf = 2000.
gpu = False
fraction_FDM = 1.0

cp = np_
if gpu:
	cp = np
else:
	np = np_
L = 20*2 / cp.sqrt(3)
dx = L/N
m22 = cp.array([5.00000001])
hbar_ = au.h_tilde(m22)

n_p = int(3e4)
M = 1e6 # plummer sphere mass
R = 0.5
initial_drop = 0
initial_time = 0.

if initial_drop != 0:
	field_dir = f'Data/{simName}/psi/drop{initial_drop}.npy'
	r_dir = f'Data/{simName}/r/drop{initial_drop}.npy'
	v_dir = f'Data/{simName}/v/drop{initial_drop}.npy'

sigma_dm = 5. * au.kms2kpcMyr
n_streams = 2048
Mtot = 1e8

print(hbar_[0] / sigma_dm**2)
# returns a random variable in a ball
def randomInBall(Npoints):  
	"""
	returns random variables in a 3 ball

	:Npoints: int, number of points to return
	"""  
	x = np.random.normal(0,1,Npoints)
	y = np.random.normal(0,1,Npoints)
	z = np.random.normal(0,1,Npoints)

	points = np.zeros((Npoints, 3))
	points[:,0] = x
	points[:,1] = y    
	points[:,2] = z

	norm  = 1./npl.norm(points, axis = 1)
	points = np.einsum("ij,i->ij",points, norm )
	#mag = np.random.exponential(size = Npoints)
	mag = sp2.maxwell.rvs(size = Npoints )
	points = np.einsum("ij,i->ij",points, mag )

	return points

def GetPsi():
	cp = np_
	if gpu:
		cp = np
	X,Y,Z = gu.grid((N,N,N),L,gpu=gpu)
	psi = cp.zeros((nf, N, N, N)) + 0j
	stream_velocities = cp.zeros((nf, n_streams,3))
	time0 = time.time()
	np.random.seed(1)

	for j in range(nf):

		v_streams = randomInBall(n_streams)*sigma_dm
		stream_velocities[j,:] = v_streams

		for i in range(n_streams):
		
			v_ = v_streams[i]
			k_ = v_ / hbar_[j]
			S_ = cp.rint(k_*L / 2 / np.pi)
			v_ = S_*2*cp.pi * hbar_[j] / L
			v_mag = cp.sqrt(cp.sum(cp.abs(v_)**2))
			w_ = 1.#np.exp(-.5*(v_mag/sigma_dm)**2)
			arg = -1j*(v_[0]*X + v_[1]*Y + v_[2]*Z) / hbar_[j]
			phi = cp.random.uniform(0,2*cp.pi)
			psi[j,:,:,:] += cp.sqrt(w_)*cp.exp(arg)*cp.exp(1j*phi)
			done = i + j*len(v_streams) + 1

			su.PrintTimeUpdate(done, nf*n_streams, time0)

		psi[j,:,:,:] /= cp.sqrt(cp.sum(cp.abs(psi[j,:,:,:])**2)*dx**3)
		psi[j,:,:,:] *= cp.sqrt(Mtot)

	return psi



def SetICs():

	cp = np_
	if gpu:
		cp = np

	psi = cp.zeros((nf,N,N,N)) + 0j
	r = cp.zeros((n_p,3))
	v = cp.zeros((n_p,3))

	if initial_drop == 0:
		### set up initial conditions
		r,v = au.PlummerSphere(M,R,n_p,gpu=gpu)
		psi = GetPsi()
	else:
		psi[0,:,:,:] = cp.load(field_dir)
		v = cp.load(v_dir)
		r = cp.load(r_dir)

	s = MS.Solver()
	### set simulation parameters
	s.SetParams(simName=simName, N = N, data_drops = data_drops, padded=padded,
	 cf=cf, L = L, m22 = m22, C = C, Tf = Tf, gpu = gpu, r = r, v = v, np = n_p,
	 mp = M/n_p)
	### set initial field
	s.D = 3

	s.initial_drop = initial_drop
	s.T_initial = initial_time
	s.explicit_particle_forces = True
	s.eps = 1e-9
	s.initial_drop = initial_drop
	s.T_initial = initial_time

	s.set_psi(psi)
	s.set_K()

	extras = {}
	extras['fraction_FDM'] = fraction_FDM
	extras['R'] = R
	extras['M'] = M
	extras['n_streams'] = n_streams
	extras['sigma_dm'] = sigma_dm
	extras['M_dm'] = Mtot
	s.extras = extras

	s.oldUpdateRule = True

	return s


if __name__ == "__main__":
	# set up sim ics
	s = SetICs()
	
	# fo = pu.FigObj()
	# fo.AddPlot(su.cpuThis(np.abs(s.psi[0,0,:,:])**2) )
	# fo.show()

	# - run sim
	s.RunSim()