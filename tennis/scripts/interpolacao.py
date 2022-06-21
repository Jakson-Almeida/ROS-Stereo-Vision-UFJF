from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x**2/9.0)
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')

xnew = np.linspace(0, 10, num=41, endpoint=True)
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()

class StereoInterpolation:
    def __init__(self):
        self.x = np.linspace(0, 9, num=10)
        self.y = np.linspace(0, 9, num=10)
        self.t = np.linspace(0, 9, num=10)
        self.a = np.linspace(0, 9, num=10)

    def update(self, x, y):
        #fila
        x = x

    def estimate(self, s0, s1):
        f1 = interp1d(self.a, self.x, kind='cubic')
        f2 = interp1d(self.a, self.y, kind='cubic')
        f3 = interp1d(self.a, self.t, kind='cubic')