# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 15:18:15 2021

@author: klein
"""
from ExternalFunctions import UnbinnedLH, BinnedLH, Chi2Regression
from iminuit import Minuit
import numpy as np

class Error(Exception):
    """Base class for other exceptions"""
    pass

class Invalid_Argument(Error):
    """Raised when an invalid argument was passed"""
    pass

def produce_hist_values(x_all, N_bins, x_range = None, only_nonzero = False):
    r"""
    Produce histogram
    
    Parameters
    --------------
    x_all: array_like
        Input data.
    N_bins: int
        defines the number of equal-width
        bins in the given range
    x_range: (float, float), optional
        Lower and upper range for the bins,
        if not provided range is (x_all.min(), x_all.max())
    only_nonzero: bool, optional
        if True return only values that are associated with bins with y!=0
    Returns
    -------
    x: array_like
        Centres of bins.
    y: array_like
        Counts
    sy: array_like
        Error on counts assuming Poisson distributed bin counts
    binwidth: float
    """
    if x_range==None:
        x_range = (x_all.min(), x_all.max())
    counts, bin_edges = np.histogram(x_all, bins=N_bins, range=x_range)
    x, binwidth = (bin_edges[1:] + bin_edges[:-1])/2, bin_edges[1]-bin_edges[0]
    y, sy = counts, np.sqrt(counts) #assume that the bin count is Poisson distributed.
    if only_nonzero:
        mask = counts>0
        x, y, sy = x[mask], y[mask], sy[mask]
    return x, y, sy, binwidth


def hist_fit(fit_func, x_all, p0, N_bins, x_range = None, fit_type = 'chi2',
             observed = True, print_level = 1): 
    r"""
    Perform fit on histogram data.
    
    Patameters
    ----------
    fit_func: callable
        The model function, func(x, ...). It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
    x_all: array_like
        Input data.
    p0: array_like
        Initial guess for the parameters.
    N_bins: int
        defines the number of equal-width
        bins in the given range
    x_range: (float, float), optional
        Lower and upper range for the bins,
        if not provided range is (x_all.min(), x_all.max())
    fit_type: str, optional
        Specifies type of fit 
        chi2 (default), 
        bllh binned likelihood 
        ullh unbinned likelihood
    observed: bool, optional
        only relevant for chi2 fit
        if True(default) divide by number of observed bins in the chi2 formule
        else divide by the predicted number of bins
    print_level: object, optional
        0: quiet
        1: print stuff at the end (default)
        2: 1+fit status during call
    
    Returns
    -------
    minuit_obj : object
        See iminuit documentation.
    x: array_like
        Centres of bins.
    y: array_like
        Counts
    sy: array_like
        Error on counts assuming Poisson distributed bin counts
    """
    
    if x_range == None:
        x_range = (x_all.min(), x_all.max())
    x, y, sy = produce_hist_values(x_all, N_bins,x_range = x_range)
    
    if fit_type == 'chi2':
        if observed:
            fit_object = Chi2Regression(fit_func, x[y>0], y[y>0], sy[y>0])
        else:
            fit_object = Chi2Regression(fit_func, x, y, sy, observed = False)
    
    elif fit_type == 'bllh':
        fit_object = BinnedLH(fit_func, x_all, bins=N_bins, bound=x_range, extended=True)
    
    elif fit_type == 'ullh':
        fit_object = UnbinnedLH(fit_func, x_all, extended=True)
    
    else:
        raise Invalid_Argument
   
    #get names of the func arguments which are passed to Minuit
    p0_names, p0_values =[],[] 
    for i in range(len(p0)):
            varname = fit_func.__code__.co_varnames[1+i]
            p0_names.append(varname)
            p0_values.append(p0[i])
    kwdarg={}#this dict is passed to Minuit
    for n,v in zip(p0_names, p0_values):
        kwdarg[n]=v
    
    minuit_obj = Minuit(fit_object, **kwdarg, pedantic=False, print_level=print_level )
    minuit_obj.migrad()
    return minuit_obj, x, y, sy
    

        