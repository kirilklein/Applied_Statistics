# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 20:21:34 2020

@author: klein
"""
import numpy as np
from iminuit import Minuit
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure
import matplotlib.pyplot as plt
from scipy import stats

def weighted_avg(mu,sig):
    """Computes weighted average given mean values and sigma"""
    return np.sum(mu/sig**2)/np.sum(1/sig**2)

def sig_weighted_avg(sig):
    return 1/np.sqrt(np.sum(1/sig**2))

def df_arr(data): return np.array(data,dtype='float32')

def fast_hist(data, bins ):
    fig, ax = plt.subplots()
    ax.hist(data, bins = bins)
    plt.show()
    
def lprint(*args,**kwargs):
    """Pretty print arguments as LaTeX using IPython display system 
    
    Parameters
    ----------
    args : tuple 
        What to print (in LaTeX math mode)
    kwargs : dict 
        optional keywords to pass to `display` 
    """
    display(Latex('$$'+' '.join(args)+'$$'),**kwargs)
    
def minuit_fit(fit_function,x,y,ey= None):   
    Ndof_calc = len(x)-2
    chi2_object = Chi2Regression(fit_function, x, y, ey) 
    minuit = Minuit(chi2_object, pedantic=False, a=2, b=10, print_level=0)  
    minuit.migrad();  # perform the actual fit
    Chi2_fit = minuit.fval # the chi2 value
    Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_calc)
    return minuit, Chi2_fit, Prob_fit