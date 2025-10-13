import scipy.io
import os, glob
import sys
import os
import matplotlib as mpl
import matplotlib.colors as mcolors

from scipy import  interpolate, constants
from scipy.differentiate import derivative

from dispersion_functions import *


FILE_PATH = './Accumulated_probe_table.csv'
N_MODES = 4 # number of eigenvalues sought for in Comsol!!
AUTO_CALCULATE_N_MODES = True # Automatically calculates the number of eigenvalues sought

wavelength = np.linspace(0.5,1.4,1000)*u.um #
wavelength0 = 1100*u.nm # pump wavelength
λ0vec = np.linspace(1000,1200,200)*u.nm

'''
    TEMPORARY VALUES, WHILE WE FIGURE COMSOL CALCULATION
'''
γ=18/(u.km*u.W) # nonlinear coefficient
P=10*u.W # pump power
L=5*u.km #waveguide length



def index_analysis(df, save_path='./'):
    neff_list = [] # list to store neff
    β2_list = [] # list to store GVD
    l0=df['% wv0 (m)'].iloc[0::N_MODES].values
    lambda_vec =l0*u.m # lambda in m
    fig,ax = plt.subplots(4,1,figsize=cm2inch(12,16),sharex=True)
    for i in range(N_MODES):
        #--------------
        neff_list.append(np.real(df['Effective mode index (1), neff Probe'].iloc[i::N_MODES].values)) # take only real part
        ng, β1, β2, D = GVDwg(lambda_vec ,neff_list[i])
        β2_list.append(β2) # we will need this for plotting later
        #--------------
        #--------------
        ax0=ax[0]
        ax0.plot(lambda_vec.to(u.um),neff_list[i],'-',label="mode {:}".format(i))
        ax0.set_ylabel('Effective index')
        ax0.grid(True)
        #---
        ax0=ax[1]
        ax0.plot(lambda_vec[:-1].to(u.um),ng,'-',label="mode {:}".format(i))
        ax0.set_ylabel('Group index')
        ax0.grid(True)
        #---
        ax0=ax[2]
        ax0.plot(lambda_vec[:-2].to(u.um),β2,'-',label="mode {:}".format(i))
        ax0.set_ylabel('GVD (β2) - $ps^2/km$')
        ax0.grid(True)
        #---
        ax0=ax[3]
        ax0.plot(lambda_vec[:-2].to(u.um),D,'-',label="mode {:}".format(i))
        ax0.set_ylabel('D - $ps/nm.km$')
        ax0.set_xlabel(r'Wavelength($\mu$m)')    
        ax0.grid(True)

    ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)    

    save_file = os.path.join(save_path, 'dispersion.png')
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()

    return neff_list, β2_list, lambda_vec


def beta2_analysis(df, β2_list, lambda_vec, save_path='./'):
    #Interpolation function for β2
    for β2 in  β2_list:
        beta2interp = interpolate.interp1d(lambda_vec[:-2], β2) #note that this has units of psˆ2/km!!
        #validating interpolation
        plt.plot(lambda_vec[:-2].to(u.um),beta2interp(lambda_vec[:-2]) )
        plt.plot(lambda_vec[:-2].to(u.um),β2,'*')
    plt.xlabel("Wavelegnth (μm)")
    plt.ylabel("β2 (ps$^2$/nm/km)")
    plt.grid()

    save_file = os.path.join(save_path, 'beta_2.png')
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


def deltabeta(wavelength, wavelength0, β2_list, lambda_vec, imode):
    #this function interpolated the Δβ function for a given pump mode frequency
    #wavelength: vector to interpolate in
    #wavelength0: pump wavelength - must be within the lambda_vec range!!
    β2=β2_list[imode-1] # here we choose the last mode, because it has the highest refractive index!
    #!!!astropy cannot attribute units to interp1d objects, but the values will be in psˆ2/km!!!!!
    beta2interp = interpolate.interp1d(lambda_vec[:-2], β2) 
    #once calculated, we can restore units!!
    beta2λp=beta2interp(wavelength0.to(u.m))*u.ps**2/u.km #note that this has units of psˆ2/km!!
    freq=2*pi*cte.c/wavelength # frequency vector
    freq0=2*pi*cte.c/wavelength0 # pump frequency
    # omega=2*np.pi*freq
    # omega0=2*np.pi*freq0
    dbeta = beta2λp*(freq-freq0)**2/2
    return dbeta


def deltabeta_analysis(wavelength, wavelength0, β2_list, lambda_vec, imode=N_MODES, save_path='./'):
    freq = cte.c/wavelength # frequency vector
    deltabeta_values = deltabeta(wavelength,wavelength0,β2_list,lambda_vec,imode)
    #--
    fig,ax = plt.subplots(1,2,figsize=cm2inch(14,6),sharey=True)
    ax0=ax[0]
    ax0.plot(wavelength.to(u.um),deltabeta_values.to(1/u.mm))
    ax0.set_xlabel("Wavelegnth (μm)")
    ax0.set_ylabel("Δβ (mm)$^{-1}$")
    ax0.grid()
    ax0=ax[1]
    ax0.plot(freq.to(u.THz),deltabeta_values.to(1/u.mm))
    ax0.set_xlabel("Frequency(THz)")
    ax0.grid()

    save_file = os.path.join(save_path, 'deltabeta.png')
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


def deltabeta_pump_analysis(wavelength, β2_list, lambda_vec, imode=N_MODES, save_path='./'):
    fig,ax = plt.subplots(1,2,figsize=cm2inch(17,6),sharey=True)
    #---
    for wavelength0 in [800,1100,1150,1200,1300]*u.nm:
        freq = cte.c/wavelength # frequency vector
        deltabeta_values = deltabeta(wavelength,wavelength0,β2_list,lambda_vec,imode)
        ax0=ax[0]
        ax0.plot(wavelength.to(u.um),deltabeta_values.to(1/u.mm))
        ax0.set_xlabel("Wavelegnth (μm)")
        ax0.set_ylabel("Δβ (mm)$^{-1}$")
        ax0.grid()
        ax0=ax[1]
        ax0.plot(freq.to(u.THz),deltabeta_values.to(1/u.mm),label="λ$_0$={:}".format(wavelength0))
        ax0.set_xlabel("Frequency(THz)")
        ax0.grid()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    save_file = os.path.join(save_path, 'deltabeta_pump.png')
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()
    

def g0(wavelength,wavelength0, β2_list, lambda_vec, γ, P, imode=N_MODES):
    Δβ = deltabeta(wavelength,wavelength0,β2_list,lambda_vec,imode).to(1/u.m) # linear phase mismatch
    return np.sqrt( -Δβ*(Δβ + 4*γ*P)+0j ) # the 0j is necessary for python to take sqrt of negative numbers
def gain(wavelength,wavelength0, β2_list, lambda_vec, γ, P, L, imode=N_MODES):
    gvec = g0(wavelength,wavelength0, β2_list, lambda_vec, γ, P, imode)
    return 1+8*(γ*P/gvec)**2*np.sinh( (gvec*L/2).value )**2


def gain_analysis(wavelength, wavelength0, β2_list, lambda_vec, γ, P, L, imode=N_MODES, save_path='./'):
    #------------
    freq = cte.c/wavelength # frequency vector
    #------------------
    #PARAMETRIC GAIN
    deltabeta_values = deltabeta(wavelength,wavelength0,β2_list,lambda_vec,imode)
    Δβ=deltabeta_values.to(1/u.m) # linear phase mismatch
    gvec=g0(wavelength,wavelength0, β2_list, lambda_vec, γ, P, imode)
    Gain= gain(wavelength,wavelength0, β2_list, lambda_vec, γ, P, L, imode)
    GaindB=10*np.log10(Gain)
    dλ = (wavelength-wavelength0).to(u.um)

    #------------------
    #PLOTTING PARAMETRIC GAIN
    #-----------------
    fig,ax = plt.subplots(3,1,figsize=cm2inch(12,15),sharex=True)
    #------------------
    ax0=ax[0]
    ax0.plot(wavelength.to(u.um),Δβ)
    ax0.axhline((-2*γ*P).to(1/u.mm).value,c='red')
    ax0.set_ylabel("Δβ (mm)$^{-1}$")
    ax0.grid()
    #------------------
    ax0=ax[1]
    ax0.plot(wavelength.to(u.um),np.real(gvec))
    ax0.set_ylabel("Gain coefficient (m)$^{-1}$")
    ax0.grid()
    #------------------
    ax0=ax[2]
    ax0.plot(wavelength.to(u.um),np.real(GaindB))
    ax0.grid()
    ax0.set_ylabel("Gain (dB)")
    ax0.set_xlabel("Wavelegnth (μm)")

    save_file = os.path.join(save_path, 'gain_wavelength.png')
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()
    
    #------------------
    #PLOTTING PARAMETRIC GAIN
    #-----------------
    fig,ax = plt.subplots(3,1,figsize=cm2inch(12,15),sharex=True)
    #------------------
    ax0=ax[0]
    ax0.plot(dλ,Δβ)
    ax0.axhline((-2*γ*P).to(1/u.mm).value,c='red')
    ax0.set_ylabel("Δβ (mm)$^{-1}$")
    ax0.grid()
    #------------------
    ax0=ax[1]
    ax0.plot(dλ,np.real(gvec))
    ax0.set_ylabel("Gain coefficient (m)$^{-1}$")
    ax0.grid()
    #------------------
    ax0=ax[2]
    ax0.plot(dλ,np.real(GaindB))
    ax0.grid()
    ax0.set_ylabel("Gain (dB)")
    ax0.set_xlabel("Δλ (λ-λ$_p$) (μm)")
    
    save_file = os.path.join(save_path, 'gain_w-wp.png')
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


def gain_pump_analysis(wavelength, wavelength0, β2_list, lambda_vec, γ, P, L, imode=N_MODES, save_path='./'):
    freq = cte.c/wavelength # frequency vector
    #-----------------
    fig,ax = plt.subplots(3,1,figsize=cm2inch(12,15),sharex=True)
    for wavelength0 in np.linspace(900,1200,5)*u.nm:
    #wavelength0 = 1200e-9*u.m # pump wavelength 
        #------------------
        #PARAMETRIC GAIN
        Δβ=deltabeta(wavelength,wavelength0,β2_list,lambda_vec,imode).to(1/u.m) # linear phase mismatch
        gvec=g0(wavelength,wavelength0, β2_list, lambda_vec, γ, P, imode)
        Gain= gain(wavelength,wavelength0, β2_list, lambda_vec, γ, P, L, imode)
        GaindB=10*np.log10(Gain)
        #------------------
        #PLOTTING PARAMETRIC GAIN
        #-----------------
        #------------------
        ax0=ax[0]
        ax0.plot(wavelength.to(u.um),Δβ)
        ax0.axhline((-2*γ*P).to(1/u.mm).value,c='red')
        ax0.set_ylabel("Δβ (mm)$^{-1}$")
        ax0.grid()
        #------------------
        ax0=ax[1]
        ax0.plot(wavelength.to(u.um),np.real(gvec),label="λ$_0$={:}".format(wavelength0))
        ax0.set_ylabel("Gain coefficient (m)$^{-1}$")
        ax0.grid()
        #------------------
        ax0=ax[2]
        ax0.plot(wavelength.to(u.um),np.real(GaindB))
        ax0.grid()
        ax0.set_ylabel("Gain (dB)")
        ax0.set_xlabel("Wavelegnth (μm)")
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) 
        
    save_file = os.path.join(save_path, 'paramteric_gain_lp_sweep.png')
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


def zero_GVD_wl(wavelength, β2_list, lambda_vec, imode=N_MODES):
    beta2interp = interpolate.interp1d(lambda_vec[:-2], β2_list[imode-1]) 
    index=np.argmin(beta2interp(wavelength.to(u.m))>0)
    return wavelength[index].to(u.nm)


def gain_density_analysis(wavelength, β2_list, λ0vec, imode=N_MODES, save_path='./'):
    beta2interp = interpolate.interp1d(lambda_vec[:-2], β2_list[imode-1]) 
    GainMatrix=np.zeros([len(wavelength),len(λ0vec)])
    for i,wavelength0 in enumerate(λ0vec):
        Gain= gain(wavelength,wavelength0, β2_list, lambda_vec, γ, P, L, imode)
        GaindB=10*np.log10(Gain)
        #----
        GainMatrix[:,i]=np.real(GaindB)


    fig,ax = plt.subplots(1,2,sharey=True, gridspec_kw={'width_ratios': [0.5, 1]})
    ax0=ax[0]
    ax0.plot(beta2interp(λ0vec.to(u.m))*u.ps**2/u.km,λ0vec.value)
    ax0.set_ylabel("Pump Wavelegnth (μm)")
    ax0.set_xlabel("β2 (ps$^2$/nm/km)")
    ax0.grid()
    ax0=ax[1]
    c = ax0.pcolormesh(wavelength.value,λ0vec.value , np.transpose(GainMatrix), cmap='plasma', vmin=GainMatrix.min(), vmax=GainMatrix.max(), rasterized=True)
    plt.colorbar(c, label='Norm. Intensity')
    ax0.grid()
    ax0.set_xlabel("Wavelegnth (μm)")
        
    save_file = os.path.join(save_path, 'parametric_gain_density_wavelength.png')
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


def gain_density_shift_analysis(wavelength, β2_list, λ0vec, imode=N_MODES, save_path='./'):
    beta2interp = interpolate.interp1d(lambda_vec[:-2], β2_list[imode-1]) 
    GainMatrix=np.zeros([len(wavelength),len(λ0vec)])
    λMatrix=np.zeros([len(wavelength),len(λ0vec)])
    λ0Matrix=np.zeros([len(wavelength),len(λ0vec)])
    for i,wavelength0 in enumerate(λ0vec):
        Gain= gain(wavelength,wavelength0, β2_list, lambda_vec, γ, P, L, imode)
        GaindB=10*np.log10(Gain)
        #----
        GainMatrix[:,i]=np.real(GaindB)
        λMatrix[:,i]=wavelength-wavelength0
        λ0Matrix[:,i]=np.ones(len(wavelength))*wavelength0


    fig,ax = plt.subplots(1,2,sharey=True, gridspec_kw={'width_ratios': [0.5, 1]})
    ax0=ax[0]
    ax0.plot(beta2interp(λ0vec.to(u.m))*u.ps**2/u.km,λ0vec.value)
    ax0.set_ylabel("Pump Wavelegnth (μm)")
    ax0.set_xlabel("β2 (ps$^2$/nm/km)")
    ax0.grid()
    ax0=ax[1]
    c = plt.pcolormesh(λMatrix,λ0Matrix, (GainMatrix), cmap='plasma', vmin=GainMatrix.min(), vmax=GainMatrix.max(), rasterized=True)
    plt.colorbar(c, label='Gain (dB)')
    ax0.grid()
    ax0.set_xlim([-0.4,0.4])
    ax0.set_xlabel("λ-λ$_p$ (μm)")
        
    save_file = os.path.join(save_path, 'parametric_gain_density_wavelength_shift.png')
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()





if __name__ == '__main__':
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['font.size'] = '12'
    rcParams['font.style'] = 'normal'
    rcParams['font.weight'] = 'medium'
    rcParams['pdf.fonttype'] = '42'

    rcParams['grid.linestyle'] = ':'
    rcParams['grid.linewidth'] = 1
    rcParams['grid.alpha'] = 0.5

    rcParams['lines.linewidth'] = 1

    # rcParams['axes.autolimit_mode'] = 'round_numbers'
    rcParams['axes.xmargin'] = 0
    rcParams['axes.ymargin'] = 0.05
    rcParams['axes.axisbelow'] = True

    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'

    plt.rcParams['svg.fonttype'] = 'none'
    rcParams["figure.frameon"] = False
    cs = list(mcolors.TABLEAU_COLORS.values())

    
    raw_df = read_comsol_table_2(FILE_PATH)
    
    D_core_values = np.unique(raw_df['D_core (m)'])

    for d_core in D_core_values:

        df = raw_df.loc[raw_df['D_core (m)'] == d_core]
        save_dir = f'./{d_core*1e6:.2f}um diameter'
        os.makedirs(save_dir, exist_ok=True)

        wv0_values = np.unique(df['% wv0 (m)'])
        if AUTO_CALCULATE_N_MODES:
            N_MODES = len(df.loc[ df['% wv0 (m)'] == wv0_values[0] ])

        neff_list, β2_list, lambda_vec = index_analysis(df, save_dir)
        zero_GVD = zero_GVD_wl(wavelength, β2_list, lambda_vec, imode=N_MODES)

        beta2_analysis(df, β2_list, lambda_vec, save_dir)

        deltabeta_analysis(wavelength, wavelength0, β2_list, lambda_vec, save_path=save_dir)
        deltabeta_pump_analysis(wavelength, β2_list, lambda_vec, save_path=save_dir)

        gain_analysis(wavelength, wavelength0, β2_list, lambda_vec, γ, P, L, imode=N_MODES, save_path=save_dir)
        gain_pump_analysis(wavelength, wavelength0, β2_list, lambda_vec, γ, P, L, imode=N_MODES, save_path=save_dir)

        gain_density_analysis(wavelength, β2_list, λ0vec, imode=N_MODES, save_path=save_dir)
        gain_density_shift_analysis(wavelength, β2_list, λ0vec, imode=N_MODES, save_path=save_dir)