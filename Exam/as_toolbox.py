# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:23:38 2021

@author: klein

Basic functions for statistics.
"""

import numpy as np
from scipy import stats

def weighted_avg(arr_mu, arr_sig):
    r"""
    Compute weighted average with uncertainty
    Parameters: 
    arr_mu, arr_sig: array_like
    Returns:
    mu: weighted average
    sig_mu: uncertainty on average
    """
    weights = 1/arr_sig**2
    mu = np.average(arr_mu, weights = weights)
    sig_mu = 1/np.sqrt(np.sum(1/arr_sig**2))
    return mu, sig_mu


def chi_sq(y_pred, y, sy):
    r"""Compute chi2. 
    Returns: sum((y_pred-y)**2/sy**2)"""
    return np.sum((y_pred-y)**2/sy**2)

def chi_sq_const(arr_mu, arr_sig):
    r"""Compute chi2 assuming y_pred=weighted_average
    Returns: sum((y_pred-y)**2/sy**2)"""
    y_pred, _ = weighted_avg(mu,sig)
    return chi_sq(y_pred, mu, sig)

def chi_sq_histtest(O1, O2): 
    r"""Compute chi2 between 2 histograms with same binning
    Parameters: O1, O2: array_like
        counts in histogram
    Returns: sum((O1-O2)**2/(O1+O2))"""
    return np.sum((O1-O2)**2/(O1+O2))    

def std_to_prob(std): 
    r"""Compute probability of having distance>std from the mean=0.
    """
    return (1-(stats.norm.cdf(abs(std))-stats.norm.cdf(-abs(std))))

def chauvenet_num(arr): 
    """Compute for each entry probability*len(arr) assuming normal distribution."""
    z = stats.zscore(arr)
    ch_num = len(arr)*std_to_prob(z)
    return ch_num

def exclude_chauvenet(arr, threshold = .5, removed = False):
    """Based on Chauvenets criterion, removes outliers from an array. 
    Parameters:
        arr: array_like, input data
        threshold: float, (default 0.5) Threshold for chauvenet criterion 
        above which points are accepted
        removed: bool, If True (default:False) return also array of removed points
    Returns: 
        Array of remaining points, array of removed points(optional)
    """
    removed = []
    while np.min((chauvenet_num(arr)))<threshold:
        min_idx = np.argmin(chauvenet_num(arr))
        removed.append(arr[min_idx])
        arr_n = np.delete(arr, min_idx)
        arr = arr_n
    if removed:
        return arr_n, removed
    else:
        return arr_n
    
def accept_reject(func, N, xmin, xmax, ymin, ymax, initial_factor = 2):
    """Produce N random numbers distributed according to the function func."""
    L = xmax-xmin
    N_i = initial_factor*N
    x_test = L*np.random.uniform(size=N_i)+xmin
    y_test = ymax*np.random.uniform(size = N_i)
    mask_func = y_test<func(x_test)
    if np.count_nonzero(mask_func)>N:
        x_func = x_test[mask_func]
        x_func = x_func[:N]
    else:
        x_func = accept_reject(xmin,xmax,ymin,ymax,func,N,initial_factor = initial_factor*2)
    return x_func
    
def transform_method(inv_func,xmin, xmax, N, initial_factor = 2): 
    r"""Produce N random numbers distr according to f(x) given the inverse 
    inv_func of
    F(x) = \int_{-inf}^x f(x')dx'"""
    N_i = 2*N
    x = inv_func(np.random.uniform(size = N_i))
    x = x[(x>xmin) & (x<xmax)]
    if len(x)>N:
        x = x[:N]
    else:
        x = transform_method(inv_func,xmin, xmax, N, initial_factor = initial_factor*2)
    return x

