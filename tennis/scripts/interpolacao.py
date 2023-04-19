from scipy.interpolate import interp1d, UnivariateSpline
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from collections import deque

# x = np.linspace(0, 10, num=11, endpoint=True)
# y = np.cos(-x**2/9.0)
# f = interp1d(x, y)
# f2 = interp1d(x, y, kind='cubic')

# xnew = np.linspace(0, 10, num=41, endpoint=True)
# plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
# plt.legend(['data', 'linear', 'cubic'], loc='best')
# plt.show()

def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))

def knots(xv, deg):
    if deg % 2 == 1:
        j = (deg+1) // 2
        interior_knots = xv[j:-j]
    else:
        j = deg // 2
        interior_knots = [Rational(a+b, 2) for a, b in zip(xv[j:-j-1], xv[j+1:-j])]
    return [xv[0]] * (deg+1) + interior_knots + [xv[-1]] * (deg+1)

class Point:
    def __init__(self, x_, y_, t_):
        self.x = x_
        self.y = y_
        self.t = t_

class StereoInterpolation:
    def __init__(self, nm = 10):
        self.c0 = deque([])
        self.c1 = deque([])
        self.t = deque([])
        self.N = np.arange(nm)
        self.n0 = 0
        for i in range(nm):
            self.c0.append(Point(0, 0, 0))
            self.c1.append(Point(0, 0, 0))
            self.t.append(Point(0, 0, 0))


    def updateC0(self, c):
        self.c0.append(c)
        self.c0.popleft()
    
    def updateC1(self, c):
        self.c1.append(c)
        self.c1.popleft()

    def estimate(self, tem):
        N0 = self.n0
        c0 = []
        c1 = []
        for obj in self.c0:
            c0.append(obj.t - tem)
        for obj in self.c1:
            c1.append(obj.t - tem)
        x = np.linspace(0, 9, num=10, endpoint=True)

        self.na = 0
        self.nb = 0

        if(c0[N0] < 0) and (c1[N0] < 0):
            for n in range(len(c0) -1 -N0):
                n = n + N0
                if(c0[n] < 0) and (c0[n+1] >= 0):
                    self.na = n
                if(c1[n] < 0) and (c1[n+1] >= 0):
                    self.nb = n
                if(self.na != 0) and (self.nb != 0):
                    break

        # print("Valores encontrados")
        # print(self.na)
        # print(self.nb)

        # print("Valor menor que 0")
        # print(c0[self.na])
        # print("Valor maior que 0")
        # print(c0[self.na + 1])

        A0 = (c0[self.na + 1] - c0[self.na])
        b0 = c0[self.na] - A0*self.na

        A1 = (c1[self.nb + 1] - c1[self.nb])
        b1 = c1[self.nb] - A1*self.nb

        # print("Debuga")
        # print(c0[self.na])
        # print(c0[self.na + 1])
        # print("")

        self.na = constrain(-b0/A0, 0, len(c0))
        self.nb = constrain(-b1/A1, 0, len(c0))

        if(np.abs(A0) < 0.000001):
            self.na = N0
        
        if(np.abs(A1) < 0.000001):
            self.nb = N0
        
        # print("Valores encontrados")
        # print(self.na)
        # print(self.nb)

        self.n0 = min(self.na, self.nb)

        xa = []
        ya = []
        xb = []
        yb = []
        for obj in self.c0:
            xa.append(obj.x)
            ya.append(obj.y)
        for obj in self.c1:
            xb.append(obj.x)
            yb.append(obj.y)

        xa = interp1d(x, xa, kind='cubic', fill_value="extrapolate")
        ya = interp1d(x, ya, kind='cubic', fill_value="extrapolate")
        xb = interp1d(x, xb, kind='cubic', fill_value="extrapolate")
        yb = interp1d(x, yb, kind='cubic', fill_value="extrapolate")

        return xa(self.na), ya(self.na), xb(self.nb), yb(self.nb)


        f0 = UnivariateSpline(x, c0, k=1)
        f1 = UnivariateSpline(x, c1, k=1)
        f0.set_smoothing_factor(0)
        f1.set_smoothing_factor(0)

        # f0 = interp1d(x, c0, kind='cubic', fill_value="extrapolate")
        # f1 = interp1d(x, c1, kind='cubic', fill_value="extrapolate")

        # print(c0)
        # print(c1)

        xnew = np.linspace(0, 9, num=81, endpoint=True)
        plt.plot(x, c0, 'o', x, c1, 'o', xnew, f0(xnew), '-', xnew, f1(xnew), '--', )
        plt.xlabel("N (amostra)")
        plt.ylabel("t (tempo)")

        # f0 = np.vectorize(f0)
        # f1 = np.vectorize(f1)

        # n0 = optimize.fsolve(f0, 0)
        # n0 = optimize.root(f0, 0)
        # n1 = optimize.root(f1, 0)
        # print("Raízes da função:")
        # print(n0.x)
        # print(n1.x)

        # print("f(x)")
        # print(f"f(n0) = {f0(n0.x)}")
        # print(f"f(n1) = {f1(n1.x)}")

        plt.show()
    
    def printC0(self):
        for ob in self.c0:
            print(f"{ob.x} {ob.y} {ob.t}")
        print(" ")

inter = StereoInterpolation()
inter.updateC0(Point(1, 1, 1.0))
inter.updateC1(Point(1, 1, 1.5))
# inter.printC0()

inter.updateC0(Point(2, 1, 1.8))
inter.updateC1(Point(1, 1.5, 2))
# inter.printC0()

inter.updateC0(Point(2, 2, 3))
inter.updateC1(Point(2, 3, 3))
# inter.printC0()

inter.estimate(1)

# fila = deque([])
# for i in range(10):
#     fila.append(Point(0, 0, 0))

# for fi in fila:
#     print(f"{fi.x} {fi.y} {fi.t}")

# fila.append(Point(10, 20, 30))
# fila.popleft()

# print("")
# for fi in fila:
#     print(f"{fi.x} {fi.y} {fi.t}")

