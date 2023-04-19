from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, num=11, endpoint=True)
x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x**2/9.0)
z = np.sin(-x**2/9.0)
f2 = interp1d(x, y, kind='cubic')
f3 = interp1d(x, z, kind='cubic')

xnew = np.linspace(0, 10, num=41, endpoint=True)
plt.plot(x, y, z, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()