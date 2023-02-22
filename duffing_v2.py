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

    This file showcases an example on a Duffing oscillator.

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as spfft
from MHBM2022 import PyHBM

def fext(omega, t): return [0.2*np.cos(omega * t)]
def fnl(q, qdot=0, omega=0, t=0): return q ** 3

problem = MHBM(fext=fext, fnl=fnl, H=9, N=37, \
               K = np.array([[1]]), M = np.array([[1]]), C=0.01*np.array([[1]]))

problem.solve(omega_start=0.2, omega_end=10, tol=0.0001, ds=0.05)

# Plot frequency response curve
omega_set = [problem.solution[i]["omega"] for i in range(problem.number_of_solutions)]
Q1_set = [problem.solution[i]["Q"][1] for i in range(problem.number_of_solutions)]
Q_set =  [problem.solution[i]["Q"] for i in range(problem.number_of_solutions)]
iter_set = [problem.solution[i]["iter"] for i in range(problem.number_of_solutions)]
plt.figure(1)
plt.plot(omega_set, np.abs(Q1_set))
print("The average number of iterations was", np.mean(iter_set))

"""# Plot time response for solution number 40
omega = problem.solution[40]["omega"]
Q = problem.solution[40]["Q"]
N = 10000
q = spfft.irfft(Q, axis=0, n=N)*(N/2)
qdot = spfft.irfft([1j*n*omega*Q[n] for n in range(problem.H+1)], axis=0, n=N)*(N/2)
plt.figure(2)
plt.plot(np.linspace(0, 2 * np.pi / omega, N, endpoint=False), q)
plt.figure(3)
plt.plot(q, qdot)"""

plt.show()