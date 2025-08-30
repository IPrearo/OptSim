import numpy as _np

from scipy.optimize import root
from math import ceil, floor

PI = 3.1415926535897932
PI2 = 6.283185307179586
C = 299792458


class MirrorGuide:
    dx: float
    dy: float
    wl: float
    k:  float

    def __init__(self, size_x, size_y, wavelength):
        self.dx = size_x
        self.dy = size_y
        self.wl = wavelength
        self.k = PI2 / wavelength

    def approx_mode_count(self):
        return _np.rint( PI * self.dx * self.dy / self.wl**2 )

class MGMode(MirrorGuide):
    mx: int
    my: int
    kx: float
    ky: float


    def __init__(self, size_x, size_y, wavelength, mx, my):
        super().__init__(size_x, size_y, wavelength)

        self.mx = mx
        self.kx = PI * mx / self.dx

        self.my = my
        self.kx = PI * mx / self.dy


class DielectricGuide:
    dx: float
    dy: float
    n_core: float
    n_clad: float
    wl: float

    def __init__(self, size_x, size_y, n_core, n_cladding, wavelength):
        self.dx = size_x
        self.dy = size_y
        self.n_core = n_core
        self.n_clad = n_cladding
        self.wl = wavelength

    @property
    def theta_c(self):
        return _np.arccos(self.n_clad / self.n_core)
        
    @property
    def NA(self):
        return _np.sqrt( self.n_core**2 - self.n_clad**2 )
    
    def approx_mode_count(self):
        return _np.rint( PI * self.dx * self.dy * (self.n_core**2 - self.n_clad**2) / self.wl**2 )

class DGMode(DielectricGuide):
    mx: int
    my: int
    kx: float
    ky: float

    def __init__(self, size_x, size_y, n_core, n_cladding, wavelength, mx, my):
        super().__init__(size_x, size_y, n_core, n_cladding, wavelength)
        self.mx = mx
        self.my = my
        
    
    def _calc_thetas(self):
        if not hasattr(self, "_consistency_condition"):
            raise AttributeError("This mode does not have a consistency condition, try a TE or TM mode?")
        
        raise NotImplementedError