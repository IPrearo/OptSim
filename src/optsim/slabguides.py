import numpy as _np

from scipy.optimize import root
from math import ceil, floor

from dataclasses import dataclass

PI = 3.1415926535897932
PI2 = 6.283185307179586
C = 299792458


class MirrorGuide:
    d: float
    wl: float
    k: float

    def __init__(self, mirror_distance, wavelength):
        self.d = mirror_distance
        self.wl = wavelength
        self.k = PI2 / self.wl

    def maximum_mode(self):
        return floor( 2 * self.d / self.wl )


class MGMode(MirrorGuide):
    m: int

    def __init__(self, mode, mirror_distance, wavelength):
        self.m = mode
        super().__init__(mirror_distance, wavelength)

    @property
    def theta(self):
        return _np.arcsin( 0.5 * self.m * self.wl / self.d )

    @property
    def ky(self):
        return self.m*PI / self.d

    @property
    def beta(self):
        return _np.sqrt( self.k**2 - ( self.m*PI / self.d )**2 )

    @property
    def group_velocity(self, n=1):
        return C/n * _np.cos( self.theta() )


class TE_MGMode(MGMode):

    def Ex(self, y, z, Am=1):
        beta = self.beta
        if self.m%2 == 1:
            am = _np.sqrt(2*self.d) * Am
            um = _np.sqrt(2/self.d) * _np.cos(self.m*PI*y / self.d)
        else:
            am = 1j*_np.sqrt(2*self.d) * Am
            um = _np.sqrt(2/self.d) * _np.sin(self.m*PI*y / self.d)

        return am*um * _np.exp(-1j * beta * z)


class TM_MGMode(MGMode):

    def Ey(self, y, z, Am=1):
        beta = self.beta
        theta = self.theta

        if self.m%2 == 1:
            am = _np.sqrt(2*self.d) * Am
            um = _np.sqrt(2/self.d) * _np.cos(self.m*PI*y / self.d)
        else:
            am = 1j*_np.sqrt(2*self.d) * Am
            um = _np.sqrt(2/self.d) * _np.sin(self.m*PI*y / self.d)

        return am*um / _np.tan(theta) * _np.exp(-1j * beta * z)
    
    def Ez(self, y, z, Am=1):
        beta = self.beta
        if self.m%2 == 1:
            am = _np.sqrt(2*self.d) * Am
            um = _np.sqrt(2/self.d) * _np.cos(self.m*PI*y / self.d)
        else:
            am = 1j*_np.sqrt(2*self.d) * Am
            um = _np.sqrt(2/self.d) * _np.sin(self.m*PI*y / self.d)

        return am*um * _np.exp(-1j * beta * z)
    


class DielectricGuide:
    wl: float
    k: float
    n_core: float
    n_clad: float
    d: float

    def __init__(self, n_core, n_clad, wavelength, wall_distance):
        self.n_core = n_core
        self.n_clad = n_clad
        self.wl = wavelength
        self.k = PI2 / wavelength
        self.d = wall_distance

    @property
    def theta_c(self):
        return _np.arccos(self.n_clad / self.n_core)
    
    @property
    def NA(self):
        return _np.sqrt( self.n_core**2 - self.n_clad**2 )
    
    def maximum_mode(self):
        return ceil( _np.sin(self.theta_c) * 2 * self.d / self.wl )


class DGMode(DielectricGuide):
    m:int

    def __init__(self, mode, n_core, n_clad, wavelength, wall_distance):
        super().__init__(n_core, n_clad, wavelength, wall_distance)
        self.m = mode
        self.theta = None

    @property
    def beta(self):
        if self.theta is None:
            raise AttributeError("This mode does not have a theta, try a TE or TM mode?")
        return self.n_core * self.k * _np.cos(self.theta)
    
    @property
    def ky(self):
        if self.theta is None:
            raise AttributeError("This mode does not have a theta, try a TE or TM mode?")
        return self.n_core * self.k * _np.sin(self.theta)
    
    
    def _calc_theta(self):
        if self.m is None:
            raise AttributeError("This mode does not have a mode number, try a TE or TM mode?")
        initial_guess = 0.5*self.wl/self.d * (self.m+0.5)
        return root(self._consistency_condition, initial_guess)['x'][0]
    

    @property
    def gamma(self):
        return _np.sqrt( self.beta**2 - self.n_clad**2 * self.k**2 )

                


class TE_DGMode(DGMode):

    def __init__(self, mode, n_core, n_clad, wavelength, wall_distance):
        super().__init__(mode, n_core, n_clad, wavelength, wall_distance)
        self.theta = self._calc_theta()


    def _consistency_condition(self, theta_test):
        if self.m % 2 == 1:
            A = 1/_np.tan(PI * self.d/self.wl * _np.sin(theta_test))
        else:
            A = _np.tan(PI * self.d/self.wl * _np.sin(theta_test))

        B = _np.sqrt( _np.sin(self.theta_c)**2 / _np.sin(theta_test)**2 - 1 )
        return A-B
    

    def prop_constants(self):
        wl, theta, d, gamma = self.wl, self.theta, self.d, self.gamma
        sintheta = _np.sin(theta)

        if self.m % 2 == 1:
            sigma = _np.power(0.5*d + \
                        -0.25*wl/(PI*sintheta)*_np.sin(PI2*sintheta*d/wl) + \
                        _np.power(_np.sin(PI*sintheta*d/wl), 2) / gamma,
                        -0.5)
            
            alpha = sigma * _np.sin(PI*sintheta*d/wl) * _np.exp(0.5*gamma*d)
        
        else:
            sigma = _np.power(0.5*d + \
                        0.25*wl/(PI*sintheta)*_np.sin(PI2*sintheta*d/wl) + \
                        _np.power(_np.cos(PI*sintheta*d/wl), 2) / gamma,
                        -0.5)
            
            alpha = sigma * _np.cos(PI*sintheta*d/wl) * _np.exp(0.5*gamma*d)

        # Proportionallity constants for inside and outside the core, respectivelly
        return sigma, alpha


    def _internal_um(self, y):
        sigma, __ = self.prop_constants()
        if self.m % 2 == 1:
            return sigma*_np.sin(PI2 * _np.sin(self.theta) / self.wl * y)
        
        return sigma*_np.cos(PI2 * _np.sin(self.theta) / self.wl * y)


    def _external_um(self, y):
        __, alpha = self.prop_constants()
        gamma = self.gamma
        if self.m % 2 == 1:
            return _np.where( y<0.5*self.d,
                            -alpha*_np.exp( -gamma *_np.abs(y) ),
                            alpha*_np.exp( -gamma *_np.abs(y) ) )
        
        return alpha*_np.exp( -gamma *_np.abs(y) )


    def _um(self, y):
        return _np.where( _np.abs(y) > 0.5*self.d,
                           self._external_um(y),
                           self._internal_um(y))


    def Ex(self, y, z, am=1):
        z_comp = _np.exp(-1j * self.beta * z)
        return am * self._um(y) * z_comp
    


class TM_DGMode(DGMode):

    def __init__(self, mode, n_core, n_clad, wavelength, wall_distance):
        super().__init__(mode, n_core, n_clad, wavelength, wall_distance)
        self.theta = self._calc_theta()

    def _consistency_condition(self, theta_test):
        if self.m % 2 == 1:
            A = 1/_np.tan(PI * self.d/self.wl * _np.sin(theta_test))
        else:
            A = _np.tan(PI * self.d/self.wl * _np.sin(theta_test))

        B = _np.sqrt( _np.sin(self.theta_c)**2 / _np.sin(theta_test)**2 - 1 )
        C = _np.cos(self.theta_c)**2
        return A - B / C
