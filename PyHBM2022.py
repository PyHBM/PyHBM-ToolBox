"""
    PyHBM is a program that uses the Multi-harmonic Balance Method for Periodic Steady-state
    Nonlinear Structural Dynamics. It also performs numerical continuation to build the frequency response.

    Copyright (C) 2022  Tiago S. Martins

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""


import numpy as np
from dual_numbers import dual
from scipy import linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import conjugate as conj
from copy import deepcopy  # copy lists avoiding python aliasing
import warnings
import scipy.fft as spfft
import time as time_py
from mpmath import findroot

INTERACTIVE_PLOT = False

if INTERACTIVE_PLOT:
    plt.ion()
    fig = plt.figure()

class PyHBM:
    """
    Class for all the variables and methods relating to Multi-Harmonic Balance Method and Continuation

    :ivar fext: external force
    :ivar fnl: non-linear force
    :ivar M: mass matrix
    :ivar K: stiffness matrix
    :ivar C: damping matrix
    :ivar name: optional name for the problem
    :ivar H: harmonic truncation order
    :ivar N: number of time samples
    :ivar d: number of spatial DoF
    :ivar A: linear dynamics matrix in the frequency domain
    :ivar dAdomega: derivative of A in order to the frequency
    :ivar L: reduction matrix for the rigid body modes
    :ivar Fext: Fourier coefficients of the external force
    :ivar size_of_R: number of DoF in total, spatial and harmonics
    :ivar f: enlarged residue
    :ivar R: residue
    :ivar gs: parameterization constraint
    :ivar f_previous: enlarged residue at the previous iteration
    :ivar X: solver variables
    :ivar Q: Fourier coefficients of the system response
    :ivar omega: frequency of the first harmonic
    :ivar Xtilde: predicted solution
    :ivar Xlast: last solution
    :ivar deltaX: increment to solution in iterative corrector step
    :ivar V: predictor vector
    :ivar Vref: reference predictor vector
    :ivar Jac: jacobian of the enlarged residue in order to the solver variables
    :ivar dRdQ: derivative of the residue in order to the Fourier coefficients of the system response
    :ivar dFnldQ: derivative of the non linear force in order to the system response (all in frequency domain)
    :ivar q: time response of the system *position*
    :ivar qdot: time response of the system *velocity*
    :ivar qdual: system *position* dual vector
    :ivar qdotdual: system *velocity* dual vector
    :ivar dfnl: derivative of the non linear force in order to the system response (all in time domain)
    :ivar dfnldq: derivative of the non linear force in order to the system *position* (all in time domain)
    :ivar dfnldqdot: derivative of the non linear force in order to the system *velocity* (all in time domain)
    :ivar E1: exponential parameter for the iDFT
    :ivar solution: list of points in the solution curve
    :ivar number_of_solutions: number of solutions found in each run of the solver
    :ivar time: sequence of sampling instants
    :ivar omega_min: minimum frequency for continuation
    :ivar omega_max: maximum frequency for continuation
    :ivar tol: maximum tolerance for residue
    :ivar iterMax: maximum number of iterations in the corrector step
    :ivar ds: predictor step size
    :ivar max_condition_number: maximum condition number of the jacobian
    :ivar parameterization: type of PC parameterization
    :ivar predictor: type of predictor
    :ivar iter: number of iterations in the current/last corrector step
    """
    def __init__(self, fext, fnl, H=1, N=3, M=np.array([[1]]), K=np.array([[1]]), C=np.array([[1]]), name="Multi-Harmonic Balance & Continuation", rigid_modes_reduction=True):

        """
        :param fext: external force
        :param fnl: non-linear force
        :param H: harmonic truncation order
        :param N: number of time samples
        :param M: mass matrix
        :param K: stiffness matrix
        :param C: damping matrix
        :param name: optional name for the problem

        :type fext: list, or class 'numpy.ndarray'
        :type fnl: list, or class 'numpy.ndarray'
        :type H: int
        :type N: int
        :type M: class 'numpy.ndarray'
        :type K: class 'numpy.ndarray'
        :type C: class 'numpy.ndarray'
        :type name: string
        """

        self.H = H
        self.N = N
        self.fext, self.fnl = fext, fnl
        self.M, self.K, self.C = M, K, C
        self.name = name

        if N <= 2*H:
            print("The Nyquist criterion is NOT satisfied for all harmonics up to truncation order since N < 2H+1 \n" +
                  "The highest safeguarded harmonic is n =", (N-1)//2)

        dd = self.M.shape

        # Safe proofing the matrices -> square and symmetric
        if dd[0] != dd[1] or dd != self.C.shape or dd != self.K.shape:
            exit('Error: all linear system matrices must be square and have the same size')

        self.M = (self.M + self.M.T) * 0.5
        self.C = (self.C + self.C.T) * 0.5
        self.K = (self.K + self.K.T) * 0.5
        self.d = dd[0]

        # test dimension of forces

        self.A = [np.zeros((self.d, self.d)) for n in range(self.H+1)]
        self.dAdomega = [np.zeros((self.d, self.d)) for n in range(self.H+1)]
        self.L = [np.eye(self.d), False, self.d]

        self.A[0] = self.K
        NULLSPACE = la.null_space(self.A[0])

        if NULLSPACE.size:
            print('Linear system has rigid body modes: \n' + str(NULLSPACE))
            if rigid_modes_reduction:
                NULLSPACE_Mnorm = np.array([NULLSPACE[:, m] / np.sqrt(np.matmul(np.matmul(conj(NULLSPACE[:, m].T), self.M),
                                                                            NULLSPACE[:, m])) for m in range(NULLSPACE.shape[1])])

                self.L[0] = la.null_space(np.matmul(NULLSPACE_Mnorm, self.M))
                self.L[1] = True
                self.L[2] = self.L[0].shape[1]
                self.A[0] = np.matmul(np.matmul(self.L[0].T, self.A[0]), self.L[0])

        self.Fext = spfft.rfft([self.fext(1, 2*np.pi*k/self.N) for k in range(self.N)], axis=0)[:self.H+1, :, np.newaxis]
        self.normfext = np.linalg.norm(self.Fext)
        self.Fext = list(self.Fext)
        self.Fext[0] = np.matmul(self.L[0].T, self.Fext[0])

        self.dAdomega[0] = np.zeros_like(self.A[0])
        self.size_of_R = self.L[2] + self.d * 2 * self.H
        self.size_of_Rplus0H = self.L[2] + self.d * self.H
        self.size_of_Rminus = self.size_of_R -self.size_of_Rplus0H

        self.id = [self.H * self.d] + [self.L[2] + (n + self.H - 1) * self.d for n in range(1, self.H + 1)] + [(self.H - n) * self.d for n in range(1, self.H + 1)]

        # INITIALIZATION OF CONSTANT DIMENSION VECTORS
        self.f = np.zeros((self.size_of_R + 1, 1), dtype=complex)
        self.f_previous = np.zeros((self.size_of_R + 1, 1), dtype=complex)
        self.X = np.zeros((self.size_of_R + 1, 1), dtype=complex)
        self.Xtilde = np.zeros_like(self.X)
        self.Xlast = np.zeros_like(self.X)
        self.deltaX = np.zeros_like(self.X)
        self.Jac = np.zeros((self.size_of_R + 1, self.size_of_R + 1), dtype=complex)

        # Alias Creation
        self.R = [[] for n in range(2*self.H+1)]
        self.R[0] = self.f[self.id[0]:self.id[1]]
        for n in range(-1*self.H, 0): self.R[n] = self.f[self.id[n]: self.id[n]+self.d]
        for n in range(1, self.H+1): self.R[n] = self.f[self.id[n]: self.id[n]+self.d]

        self.gs = self.f[-1:]
        self.dRdQ = self.Jac[:-1, :-1]
        self.dRplusdQ = self.dRdQ[self.id[0]:, self.id[0]:]
        self.dRminusdQ = self.dRdQ[:self.id[0], self.id[0]:]
        self.dRplusdQplus = self.dRdQ[self.id[1]:, self.id[1]:]
        self.dRminusdQplus = self.dRdQ[:self.id[0], self.id[1]:]
        self.dRplusdQminus = self.dRdQ[self.id[1]:, :self.id[0]]
        self.dRminusdQminus = self.dRdQ[:self.id[0], :self.id[0]]
        self.dR0dQminus = self.dRdQ[self.id[0]:self.id[1], :self.id[0]]
        self.dR0dQplus = self.dRdQ[self.id[0]:self.id[1], self.id[1]:]
        self.dRdomega = self.Jac[:-1, -1:]
        #self.dFnldQ = [[np.zeros((self.d,self.d), dtype=complex) for n in range(self.H+1)] for k in range(self.N)]

        self.Q = [None for n in range(self.H + 1)]
        self.Q[0] = self.X[self.id[0]:self.id[1]]
        for n in range(1, self.H + 1): self.Q[n] = self.X[self.id[n]: self.id[n]+self.d]

        self.q = np.zeros((self.N,self.d))
        self.qdot = np.zeros((self.N,self.d))
        self.V = np.eye(1, self.size_of_R + 1, self.size_of_R).T  # [0,...,0,1].T
        self.Jac[-1:, :] = self.V.T # maintain throughout the correction step
        self.omega = np.array([1])
        self.omega = self.X[-1:]

        # INITIALIZATION OF CONSTANT DIMENSION DUAL VECTORS
        self.qdual = [np.array([dual(None, np.eye(1, 2*self.d, p).ravel()) for p in range(self.d)]) for k in range(self.N)]
        self.qdotdual = [np.array([dual(None, np.eye(1, 2*self.d, p).ravel()) for p in range(self.d,2*self.d)]) for k in range(self.N)]

        # build qdual and qdotdual  as alias of q and qdot
        for k in range(self.N):
            for p in range(self.d):
                self.qdual[k][p].x = self.q[k][p:p+1]
                self.qdotdual[k][p].x = self.qdot[k][p:p+1]

        # INITIALIZATION OF MATRIX FORM DERIVATIVES OF CONSTANT DIMENSION
        self.dfnl = [[np.zeros((1, 2 * self.d)).ravel() for p in range(self.d)] for k in range(self.N)]
        self.dfnldq =  [[None for p in range(self.d)] for k in range(self.N)]
        self.dfnldqdot = [[None for p in range(self.d)] for k in range(self.N)]

        # create alias for fnl and fext derivatives
        for k in range(self.N):
            for p in range(self.d): # create alias: each variable has to be a list/array
                self.dfnldq[k][p] = self.dfnl[k][p][:self.d]
                self.dfnldqdot[k][p] = self.dfnl[k][p][self.d:2*self.d]

        self.E1 = np.exp(2 * np.pi * 1j * np.matmul(np.arange(self.H + 1)[:,np.newaxis], np.arange(self.N)[np.newaxis,:]) / self.N) / self.N
        #self.C1 = np.cos(2 * np.pi * np.matmul(np.arange(self.N)[:, np.newaxis], np.arange(self.H + 1)[np.newaxis, :]) / self.N) / self.N
        #self.S1 = np.sin(2 * np.pi * np.matmul(np.arange(self.N)[:, np.newaxis], np.arange(self.H + 1)[np.newaxis, :]) / self.N) / self.N

        self.solution = []

        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
        warnings.filterwarnings("ignore", category=np.ComplexWarning)

    def build_lin_sys(self):

        for n in range(1,self.H+1):
            self.A[n]= self.K + 1j*n*self.omega*self.C - ((n*self.omega)**2)*self.M
            #if la.null_space(self.A[n]).size:
            #    print(
            #        'Linear System is singular at the base frequency of %f Hz for the harmonic n = %d and might exhibit unbounded behaviour, ie, resonance' % (
            #        self.omega[0] / (2 * np.pi), n))
            #    exit('Provide a linear system without flexible modes for the given frequency range and corresponding relevant harmonic multiples')

            self.dAdomega[n]= 1j*n*self.C - 2*self.omega*(n**2)*self.M

    def recompute_frequency_dependant_parameters(self):
        self.time = np.linspace(0, 2 * np.pi / self.omega, self.N, endpoint=False)
        self.build_lin_sys()

    def compute_residue(self):

        # build q and qdot
        self.q[:] = spfft.irfft([np.matmul(self.L[0], self.Q[0])] + self.Q[1:], self.N, axis=0)\
            .reshape((self.N, self.d))
        self.qdot[:] = spfft.irfft([np.zeros((self.d, 1))] + [1j * n * self.omega * self.Q[n]\
            for n in range(1, len(self.Q))], self.N,axis=0).reshape((self.N, self.d))

        self.Fnl = list(spfft.rfft(
            [self.fnl(q=self.q[k], qdot=self.qdot[k], t=self.time[k], omega=self.omega)[:, np.newaxis] for k in
             range(self.N)], axis=0)[:self.H + 1])

        self.R[0][:] = np.matmul(self.A[0], self.Q[0]) + np.matmul(self.L[0].T, self.Fnl[0]) - self.Fext[0]
        for n in range(1, self.H + 1):
            self.R[n][:] = np.matmul(self.A[n], self.Q[n]) + (self.Fnl[n] - self.Fext[n])
            self.R[-1*n][:] = np.conjugate(self.R[n])  # R_{N-n} = conj(R_{n})

    def compute_exact_jacobian(self):

        for k in range(self.N):
            fnl_k = self.fnl(q=self.qdual[k], qdot=self.qdotdual[k], t=self.time[k,0], omega=self.omega[0])
            for p in range(self.d):
                self.dfnl[k][p][:] = fnl_k[p].dx

        self.dFnldQ = spfft.fft([[(np.array(self.dfnldq[k]) + 1j * n * self.omega * np.array(self.dfnldqdot[k])) *
                                   self.E1[n, k] for n in range(self.H+1)] for k in range(self.N)], axis=0)
        self.dFnldQ = np.concatenate((self.dFnldQ[:self.H+1], self.dFnldQ[-1*self.H:]), axis=0)

        if self.L[1]:
            for n in range(self.H+1): # dFnldQ follows FFT indexing for axis=0 and follows Q indexing for axis=1
                self.dFnldQ[0,n] = np.matmul(self.L[0].T, self.dFnldQ[0,n])
                self.dFnldQ[n,0] = np.matmul(self.dFnldQ[n,0], self.L[0])
            for n in range(self.H + 1, 2 * self.H + 1): self.dFnldQ[n,0] = np.matmul(self.dFnldQ[n,0], self.L[0])

        self.dRplusdQ[:,:] = la.block_diag(*self.A) + self.dFnldQ[:self.H+1].transpose(0,2,1,3).reshape(self.size_of_Rplus0H,self.size_of_Rplus0H)
        self.dRminusdQ[:,:] = self.dFnldQ[-1:self.H:-1].transpose(0,2,1,3).reshape(self.size_of_Rminus,self.size_of_Rplus0H)
        self.dRminusdQminus[:,:] = conj(self.dRplusdQplus)
        self.dRplusdQminus[:, :] = conj(self.dRminusdQplus)
        self.dR0dQminus[:,:] = conj(self.dR0dQplus)

        #self.dFdomega = np.fft.rfft([np.matmul(self.dfnldqdot[k], self.qdot[k])/self.omega for k in range(self.N)], axis=0) #####
        #self.dRdomega[self.id[0]:self.id[1],:] = np.matmul(self.L[0].T, self.dFdomega[0]) #####
        self.dRdomega[self.id[1]:,:] = np.array([(np.matmul(self.dAdomega[n], self.Q[n])).ravel() for n in range(1, self.H+1)]).ravel()[:,np.newaxis] ##+ self.dFdomega[n]
        self.dRdomega[self.id[-1]:self.id[0], :] = conj(self.dRdomega[self.id[1]:,:])

    def broyden_update(self):
        self.Jac += np.matmul(self.f -self.f_previous - np.matmul(self.Jac, self.deltaX), self.deltaX.T) / ( norm(self.deltaX) ** 2)

    def first_point(self):

        # implement homotopy with damping

        self.time = np.linspace(0, 2 * np.pi / self.omega, self.N, endpoint=False)
        self.build_lin_sys()

        for self.iter in range(self.iterMax):

            self.compute_residue()

            normf = norm(self.f[:-1,:])
            if normf < self.tol * self.normfext:
                #print("Newton-Raphson: bellow absolute tolerance (%f < %f) at iteration number %d\n-> Convergence reached" % (normf / self.normfext, self.tol, self.iter))
                break

            # test convergence to choose derivative updating method
            self.compute_exact_jacobian()
            # self.broyden_update()
            # test condition number
            if abs(np.linalg.cond(self.Jac[:-1,:-1])) > self.max_condition_number:
                print("Newton-Raphson:  the current point is near stationary -> convergence compromised")
                break
            self.deltaX[:-1,:] = -la.solve(self.Jac[:-1,:-1], self.f[:-1,:])
            self.X[:-1,:] += self.deltaX[:-1,:]

        else:
            print("Newton-Raphson: maximum number of iterations exceeded -> %d\nLatest difference: max(abs(residue)) = %e" % (
                self.iterMax, np.max(abs(self.f[:,-1] / self.N))))

    def corrector_step(self):

        for self.iter in range(self.iterMax):

            self.recompute_frequency_dependant_parameters()

            self.compute_residue()

            if self.parameterization == 'orthogonal': # UPDATE RESIDUE FOR ORTHOGONAL PARAMETERIZATION
                    self.gs[:] = np.vdot(self.V, self.X - self.Xtilde)

            elif self.parameterization == 'arc length': # UPDATE RESIDUE AND GRADIENT FOR ARC LENGTH PARAMETERIZATION
                    self.gs[:] = norm(self.X - self.Xlast) ** 2 - self.ds ** 2
                    self.Jac[-1:, :] = 2 * conj(self.X - self.Xlast).T

            normf = norm(self.f)
            if normf < self.tol * self.normfext:
                #print("Newton-Raphson: bellow absolute tolerance (%f < %f) at iteration number %d\n-> Convergence reached" % (normf / self.normfext, self.tol, self.iter))
                break

            # test convergence to choose derivative updating method
            self.compute_exact_jacobian()
            # self.broyden_update()
            # test condition number
            if abs(np.linalg.cond(self.Jac)) > self.max_condition_number:
                print("Newton-Raphson:  the current point is near stationary -> convergence compromised")
                break
            self.deltaX = -la.solve(self.Jac, self.f)
            self.X += self.deltaX
            self.X *= conj(self.X[-1,0])/abs(self.X[-1,0])

        else:
            print("Newton-Raphson: maximum number of iterations exceeded -> %d\nLatest difference: max(abs(residue)) = %e" % (
                self.iterMax, np.max(abs(self.f[:,-1] / self.N))))

        print("", end="\r")
        if self.omega_max != self.omega_min:
            progress = np.real(100*(self.omega-self.omega_min)/(self.omega_max-self.omega_min)) #np.real(100*(np.log(self.omega/self.omega_min))/(np.log(self.omega_max/self.omega_min)))
            #print("*" * int(progress) + "." * int(100 - progress) + " %.3f %%" % (progress), end="")
            print(" %.3f %% " % (progress), end="")

    def predictor_step(self):

        # NORMALIZE Vref
        self.Vref /= norm(self.Vref)

        if self.predictor == 'tangent': # TANGENT PREDICTOR STEP
                self.compute_exact_jacobian()
                self.V = np.array(la.null_space(self.Jac[:-1, :]))
                if self.V[-1, 0] == 0:
                    self.V *= np.real(np.vdot(self.Vref, self.V)) / (
                                norm(self.V) * abs(np.real(np.vdot(self.Vref, self.V))))
                else:
                    self.V *= conj(self.V[-1, 0]) * np.real(
                        np.vdot(self.Vref, self.V) * conj(self.V[-1, 0])) / (norm(self.V) * abs(self.V[-1, 0] * np.real(
                        np.vdot(self.Vref, self.V) * conj(self.V[-1, 0]))))
                self.Xtilde = self.X + self.ds * self.V

        elif self.predictor == 'secant': # SECANT PREDICTOR STEP
                self.Xtilde = self.X + self.ds * self.Vref

        self.Xlast = np.array(self.X)
        self.X[:,:] = np.array(self.Xtilde)

    def solve(self, omega_start: float, omega_end: float =-1, tol=0.00001, iterMax=200, ds=0.1, max_condition_number=10**20, parameterization = 'orthogonal', predictor = 'tangent'):
        """

        :param omega_start: frequency at the start of the continuation path
        :param omega_end: frequency at the end of the continuation path
        :param tol: maximum tolerance for the residue
        :param iterMax: maximum number of iterations
        :param ds: base line predictor step size
        :param max_condition_number: maximum condition number for the jacobian
        :param parameterization: choice of parameterization type for the corrector step
        :param predictor: choice of predictor type

        :type omega_start: int
        :type omega_end: int
        :type tol: float
        :type iterMax: int
        :type ds: float
        :type max_condition_number: float
        :type parameterization: string
        :type predictor: string
        """
        # START SOLVER INITIALIZATION

        if omega_end < 0 : omega_end = omega_start

        if omega_start <= omega_end:
            self.Vref = np.eye(1, self.size_of_R + 1, self.size_of_R).T
            self.omega_min = omega_start
            self.omega_max = omega_end
        else:
            self.Vref = -1 * np.eye(1, self.size_of_R + 1, self.size_of_R).T
            self.omega_min = omega_end
            self.omega_max = omega_start

        self.tol = tol
        self.iterMax = iterMax
        self.ds = (self.omega_max-self.omega_min)/50
        self.max_condition_number = max_condition_number

        if (parameterization != 'orthogonal' and parameterization != 'arc length'):
            print('The available options for parameterization are: \'orthogonal\' and \'arc length\'')
            exit('Invalid parameterization')
        self.parameterization = parameterization

        if (predictor != 'tangent' and predictor != 'secant'):# and  predictor != 'tangent+secant'):
            print('The available options for predictors are: \'tangent\' and \'secant\'') # and \'tangent+secant\'')
            exit('Invalid predictor')
        self.predictor = predictor

        # RESET
        self.X*=0
        self.Jac*=0
        self.f*=0
        self.omega[:] = omega_start

        # END SOLVER INITIALIZATION

        # FIRST POINT - LOCAL PARAMETERIZATION AND FORCE TANGENT PREDICTOR
        self.first_point()

        if self.predictor == 'secant': # REDEFINE Vref SUCH AS TO FORCE TANGENT STEP

            self.compute_exact_jacobian()
            self.V = np.array(la.null_space(self.Jac[:-1, :]))
            if self.V[-1, 0] == 0:
                self.V *= np.real(np.vdot(self.Vref, self.V)) / (
                        norm(self.V) * abs(np.real(np.vdot(self.Vref, self.V))))
            else:
                self.V *= conj(self.V[-1, 0]) * np.real(
                    np.vdot(self.Vref, self.V) * conj(self.V[-1, 0])) / (norm(self.V) * abs(self.V[-1, 0] * np.real(
                    np.vdot(self.Vref, self.V) * conj(self.V[-1, 0]))))
            self.Vref = self.V

        self.solution.append({"Q": deepcopy(self.X[self.d*self.H:-1,:]).reshape((self.H+1, self.d))/(self.N/2), "omega": deepcopy(np.real(self.omega[0])), "iter": deepcopy(self.iter)})

        while self.omega <= self.omega_max and self.omega >= self.omega_min:

            self.predictor_step()

            if self.parameterization == 'orthogonal': # PRECOMPUTE CONSTANT ROW OF JACOBIAN
                #self.V *= selg.gamma/self.ds
                self.Jac[-1:, :] = conj(self.V.T)

            self.corrector_step()
            self.X[-1,:] = np.real(self.X[-1,:]) # omega must be a real number

            # IF THERE ARE MORE THAN ONE SOLUTION POINTS Vref IS THE SECANT
            self.Vref = self.X - self.Xlast

            # SAVE SOLUTION POINT
            self.solution.append({"Q":deepcopy(self.X[self.d*self.H:-1,:]).reshape((self.H+1, self.d))/(self.N/2), "omega":deepcopy(np.real(self.omega[0])), "iter":deepcopy(self.iter)})

            if abs(np.linalg.cond(self.Jac)) > self.max_condition_number:
                print("Over maximum condition number -> the current point might be near a singularity")
                break

            # STEP SIZE ADAPTATION
            self.ds *= (3/(self.iter+1))**2 # the desired number of iterations is 2
            self.ds = max(ds/5, min(self.ds,ds*5))

        self.number_of_solutions = self.solution.__len__()
        print("", end="\r")
        print("instructions of what is in the solution and how to obtain it")