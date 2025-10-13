import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit  
from pylab import *
from astropy import constants as cte
from astropy import units as u


def read_comsol_table(full_path,column_names=[]):
    '''
    Function to read comsol tables output from "Accumulated probe tables"
    '''
    #reading header line:
    f = open(full_path,"r")
    stream = f.readlines()
    f.close()
    header_row = stream[4]
    # print('comsol header row: ',header_row)
    df = pd.read_csv(full_path,skiprows=5,header=None)
    if len(column_names)!=0:
        df.columns=column_names
    #fix complex
    for string in df.columns:
        if type(df[string][0])==type(''):
            df[string]=df[string].str.replace('i','j').values.astype('str').astype(np.complex)
    return df

def read_comsol_table_2(full_path):
    '''
    Function to read comsol tables output from "Accumulated probe table"
    It gives the tables with the column header names set up in Comsol
    '''
    #read header line:
    f = open(full_path,"r")
    stream = f.readlines()
    f.close()
    header_row = stream[4]
    # print('comsol header row: ',header_row)
    #read csv file:
    df = pd.read_csv(full_path,skiprows=4,header=0)
    #fix complex
    for string in df.columns:
        if type(df[string][0])==type(''):
            df[string]=df[string].str.replace('i','j').values.astype('str').astype(complex)
    return df

def sel_mode(df,imode=1,label='lambda0'):
    '''
    Function to select mode from Comsol parametric sweep of modal analysis problem.
    imode (integer >=1): which mode to select from dataframe
    lambda0': which variable to return
    '''
    df_mode = df.copy()
    df_mode = df.groupby([label]).head(imode).drop_duplicates([label],keep='last')
    return df_mode

def sel_mode_pol(df,imode=1,label1='lambda0',label2='frac_TM'):
    '''
    Function to select the TE/TM modes from Comsol parametric sweep of modal analysis problem.
    imode (integer >=1): which mode to select from dataframe
    'lambda0': which variable to return
    select 'frac_TM'('frac_TE') to keep the TM(TE) modes 
    '''
    df_mode = df.copy()
    df_mode = df.groupby([label1]).head(imode)
    df_mode = df_mode.drop(df_mode[df_mode[label2]<0.5].index)
    return df_mode



def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
    
def fit(function, x, y, p0=None, plot=False):
    '''
    Fits the provided function to the provided data
    returns the fitted curve evaluated at the provided xdata 
    '''   
    if p0 is None:
        best_vals, covar = curve_fit(function, x, y, maxfev=1000)
    else:
        best_vals, covar = curve_fit(function, x, y, p0, maxfev=1000)
    
    #---
    sd = np.sqrt(np.diag(covar))
    psd = sd/best_vals
    yfit = function(x,*best_vals)
    
    #---
    print('best_vals: {}'.format(best_vals))
    print('standard deviation: {}'.format(sd))
    print('percentage standard deviation: {}'.format(psd))

    #---
    if plot==True:
        fig = plt.figure(facecolor='w')
        plt.scatter(x,y, s=50, marker='o', facecolor='None', edgecolor='black')
        plt.plot(x,yfit,'-', c='red', lw=3)
    plt.show()
        
    #---
    # fitting statistics
    # squared error from the fitted curve
    devfit = (y - yfit)
    sefit = (devfit**2).sum()

    # squared error from the average ydata
    yav = y.mean()
    devav = (y - yav)
    seav = (devav**2).sum()

    # R squared
    rsquared = 1-sefit/seav
    print('R squared:', rsquared)
    print('\n')
    
    return best_vals



def Sellmeier0(x, a0, lambda0):
    return sqrt(1+(a0*x**2)/(x**2-lambda0**2))

def Sellmeier1(x, a0, lambda0, a1, lambda1):
    return sqrt(1+(a0*x**2)/(x**2-lambda0**2)+(a1*x**2)/(x**2-lambda1**2))

def Sellmeier2(x, a0, lambda0, a1, lambda1, a2, lambda2):
    return sqrt(1+(a0*x**2)/(x**2-lambda0**2)+(a1*x**2)/(x**2-lambda1**2)+(a2*x**2)/(x**2-lambda2**2))

def GVDwg(lambda_vec,neff):
    '''
    Calculate β1, β2 and other properties
    '''
    dλ = (lambda_vec[1]-lambda_vec[0]).to(u.m)
    #--
    ng = neff[:-1]-lambda_vec[:-1]*np.diff(neff)/dλ
    dngdλ = -lambda_vec[:-2]*np.diff(neff,2)/dλ**2
    #--
    vg = cte.c/ng
    β1 = ng/cte.c
    β2 = -lambda_vec[:-2]**2/(2*np.pi*cte.c**2)*dngdλ
    #-----
    π = np.pi
    D = -2*π*cte.c/(lambda_vec[:-2]**2)*β2
    
    return ng,β1.to(u.s/u.m),β2.to(u.ps**2/u.km),D.to(u.ps/u.nm/u.km)