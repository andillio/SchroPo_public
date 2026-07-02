# pylint: disable=C,W
# Vectorised star-star self-gravity.
#
# Subclasses meshSolver.Solver and overrides ONLY ExplicitParticleForces,
# replacing the per-particle Python loop (baseSolver.ExplicitParticleForces)
# with a chunked, fully-vectorised pairwise force. This is numerically
# identical to the original but actually accelerates on the GPU.
#
# Usage: in your driver, import this in place of meshSolver, e.g.
#     import Solvers.mesh_solver_vect as MS
#     s = MS.Solver()
import numpy as np_
import meshSolver as MS
CUPY_IMPORTED = True
try:
	import cupy as cp
except ImportError:
	CUPY_IMPORTED = False


class Solver(MS.Solver):

	# Max number of particles processed per chunk. Bounds the peak memory of the
	# (chunk, n_p, 3) pairwise array (~chunk*n_p*3*itemsize). Lower if you hit
	# GPU out-of-memory; raise for a bit more speed if memory allows.
	force_chunk = 2048


	def ExplicitParticleForces(self):
		"""
		Vectorised drop-in for baseSolver.ExplicitParticleForces (3D).

		acc[i] = C/(4*pi) * sum_{j} mp[j] * (r[j]-r[i]) / (|r[j]-r[i]| + eps)^3

		The self term (j == i) has r[j]-r[i] == 0 and so contributes nothing,
		matching the original which simply excluded j == i.

		:return: array-like, [np, D] acceleration
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		# Only the 3D case is vectorised here; anything else defers to the
		# original implementation so behaviour is unchanged.
		if self.D != 3:
			return super().ExplicitParticleForces()

		r = self.r
		n = self.np

		mp = self.mp
		if isinstance(mp, (float, int)):
			mp = np.ones(n) * self.mp


		acc = np.zeros((n, 3))
		L = self.L

		for a in range(0, n, self.force_chunk):
			b = min(a + self.force_chunk, n)

			# diff[i, j] = r[j] - r[i]   (shape: chunk, n, 3)
			diff = r[np.newaxis, :, :] - r[a:b, np.newaxis, :]

			if self.mod_positions:
				diff[diff > L / 2.] -= L
				diff[diff < -1 * L / 2.] += L

			# |r[j]-r[i]| + eps   (shape: chunk, n); d*d*d avoids the slow pow()
			dist = np.sqrt(np.sum(diff * diff, axis=2)) + self.eps
			dist3 = dist * dist * dist
			weight = mp[np.newaxis, :] / dist3              # (chunk, n)

			acc[a:b] = self.C * np.sum(
				weight[:, :, np.newaxis] * diff, axis=1) / (4 * np.pi)
			
		return acc
