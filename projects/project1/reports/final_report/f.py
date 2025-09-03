from math import atan2
a = 1698.0 + 7.5033 * 1j
A = abs(a)
phi = atan2(a.imag, a.real)
print(round(A))
print(round(phi))