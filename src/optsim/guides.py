import numpy as _np

from dataclasses import dataclass

PI = 3.1415926535897932
PI2 = 6.283185307179586
C = 299792458



class MirrorGuide2D:
    d: float
    wl: float
    k: float

    def __init__(self, mirror_distance, wavelength):
        self.d = mirror_distance
        self.wl = wavelength
        self.k = PI2 / self.wl

    def maximum_mode(self):
        return int( 2 * self.d / self.wl )


class MGMode(MirrorGuide2D):

    def __init__(self, mode, mirror_distance, wavelength):
        self.m = mode
        super().__init__(mirror_distance, wavelength)

    def theta(self):
        return _np.arcsin( 0.5 * self.m * self.wl / self.d )

    def ky(self):
        return self.m*PI / self.d

    def beta(self):
        return _np.sqrt( self.k**2 - ( self.m*PI / self.d )**2 )

    def group_velocity(self, n=1):
        return C/n * _np.cos( self.theta() )


class TE_MGMode(MGMode):

    def Ex(self, y, z, Am=1):
        beta = self.beta()
        if self.m%2 == 1:
            am = _np.sqrt(2*self.d) * Am
            um = _np.sqrt(2/self.d) * _np.cos(self.m*PI*y / self.d)
        else:
            am = 1j*_np.sqrt(2*self.d) * Am
            um = _np.sqrt(2/self.d) * _np.sin(self.m*PI*y / self.d)

        return am*um * _np.exp(-1j * beta * z)


class TM_MGMode(MGMode):

    def Ey(self, y, z, Am=1):
        beta = self.beta()
        theta = self.theta()

        if self.m%2 == 1:
            am = _np.sqrt(2*self.d) * Am
            um = _np.sqrt(2/self.d) * _np.cos(self.m*PI*y / self.d)
        else:
            am = 1j*_np.sqrt(2*self.d) * Am
            um = _np.sqrt(2/self.d) * _np.sin(self.m*PI*y / self.d)

        return am*um / _np.tan(theta) * _np.exp(-1j * beta * z)
    
    def Ez(self, y, z, Am=1):
        beta = self.beta()
        if self.m%2 == 1:
            am = _np.sqrt(2*self.d) * Am
            um = _np.sqrt(2/self.d) * _np.cos(self.m*PI*y / self.d)
        else:
            am = 1j*_np.sqrt(2*self.d) * Am
            um = _np.sqrt(2/self.d) * _np.sin(self.m*PI*y / self.d)

        return am*um * _np.exp(-1j * beta * z)