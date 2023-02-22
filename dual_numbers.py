"""
    This file describes the dual Python class. It defines the specific
    rules of arithmetic and function evaluation used to perform forward
    automatic differentiation.

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

class dual:
    def __init__(self, x ,dx=np.array([1])):
        self.x=x
        self.dx=dx

    def __str__(self):
        return str(self.x) + " + " +str(self.dx) + " eps"

    def __add__(self, other):
        if isinstance(other, dual):
            return dual(self.x+other.x, self.dx+other.dx)
        else:
            return dual(self.x+other, self.dx)

    def __radd__(self, other):
        return dual(self.x+other, self.dx)

    def __sub__(self, other):
        if isinstance(other, dual):
            return dual(self.x-other.x, self.dx-other.dx)
        else:
            return dual(self.x-other, self.dx)

    def __rsub__(self, other):
        return dual(other-self.x, -1*self.dx)

    def __mul__(self, other):
        if isinstance(other, dual):
            return dual(self.x*other.x, self.x*other.dx+self.dx*other.x)
        else:
            return dual(self.x*other, self.dx*other)

    def __rmul__(self, other):
        return dual(self.x*other, self.dx*other)

    def exp(self):
        return dual(np.exp(self.x),np.exp(self.x)*self.dx)

    def __pow__(self, power, modulo=None):
        return dual(self.x**power, power*(self.x**(power-1))*self.dx)

    def __rpow__(self, base):
        return dual.exp(self*np.log(base))

    def __truediv__(self, other):
        if isinstance(other, dual):
            return dual(self.x/other.x, (self.dx*other.x-self.x*other.dx)/(other.x**2))
        else:
            return dual(self.x/other, self.dx/other)

    def __rtruediv__(self, other):
        return dual(other/self.x, (self.x-self.dx*other)/(self.x**2))

    def cos(self):
        return dual(np.cos(self.x), -1*np.sin(self.x)*self.dx)

    def sin(self):
        return dual(np.sin(self.x), np.cos(self.x)*self.dx)

    def tan(self):
        return dual(np.tan(self.x), self.dx/(np.cos(self.x)**2))

    def log(self):
        return dual(np.log(self.x), self.dx/self.x)

    def arctan(self):
        return dual(np.arctan(self.x), self.dx/(self.x**2 + 1))

    def tanh(self):
        return dual(np.tanh(self.x), self.dx/(np.cosh(self.x)**2))

    def __abs__(self):
        return dual(abs(self.x), self.dx*np.sign(self.x))

    def __lt__(self, other):
        if isinstance(other, dual):
            return self.x<other.x
        else:
            return self.x<other

    def __gt__(self, other):
        if isinstance(other, dual):
            return self.x > other.x
        else:
            return self.x > other

    def sign(self):
        return dual(np.sign(self.x), 0*self.dx)