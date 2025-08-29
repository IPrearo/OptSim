import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, brentq

wv = 1.55 # Wavelength [um]
k0 = 2 * np.pi / wv # Free space wavenumber [um^-1]
d = 5 # Core thickness [um]
n2 = 1.0 # Refractive index of cladding
n1 = 1.5 # Refractive index of core
m = 10 # Mode order considered

theta_c_bar = np.arccos(n2/n1)
theta_m_pts = np.linspace(0.01, np.pi/2 - 0.01, 200) # theta_m points [rad]
r_m_pts = np.sin(theta_m_pts) # r_m points

f1_pts = np.tan(np.pi * d * r_m_pts / wv - np.pi * m / 2)
f2_pts = np.sqrt(np.sin(theta_c_bar)**2 / r_m_pts**2 - 1)

# fig, ax = plt.subplots(layout='constrained')
# ax.set_xlabel(r'$\sin(\theta_m)$', fontsize=14)
# ax.set_ylabel(r'$f_j(\sin(\theta_m))$', fontsize=14)
# ax.set_xlim(-0.01, 1 + 0.01)
# ax.set_ylim(-10, 10)
# ax.scatter(r_m_pts, f1_pts, c='blue', s = 5, label=r'$f_1$')
# ax.scatter(r_m_pts, f2_pts, c='green', s = 5, label=r'$f_2$')
# ax.legend(loc='upper right', fontsize=12)
# plt.show()

# fig, ax = plt.subplots(layout='constrained')
# ax.set_xlabel(r'$\sin(\theta_m)$', fontsize=14)
# ax.set_ylabel(r'$f_j(\sin(\theta_m))$', fontsize=14)
# ax.set_xlim(-0.01, 1 + 0.01)
# ax.set_ylim(-10, 10)
# ax.scatter(r_m_pts, f1_pts - f2_pts, c='k', s = 5)
# plt.show()

f = lambda r_m: np.tan(np.pi*(d * r_m/wv - m/2)) - np.sqrt(np.sin(theta_c_bar)**2/r_m**2 - 1)
df = lambda r_m: np.pi * d * np.sqrt(1 - r_m**2) / (wv * np.cos(np.tan(np.pi * (d * r_m / wv - m/2)))**2) + np.sin(theta_c_bar)**2 * np.sqrt(1 - r_m**2) / (r_m**3 * np.sqrt(np.sin(theta_c_bar)**2/r_m**2 - 1))

root = brentq(f, 0.1, 0.15)
# root = newton(f, 0.1, df)
print(f"Root found: {root:.6f}")


# print(f"Root found: {root:.6f}")