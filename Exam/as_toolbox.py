# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:23:38 2021

@author: klein

Basic functions for statistics.
"""

import numpy as np
from scipy import stats
import urllib
import matplotlib.pyplot as plt
from ExternalFunctions import Chi2Regression
from ExternalFunctions import nice_string_output
from iminuit import Minuit

# In[Basic]
def load_url_data(url): 
    """Load text data under url into numpy array."""
    file= urllib.request.urlopen(url)
    return np.loadtxt(file)

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
    y_pred, _ = weighted_avg(arr_mu, arr_sig)
    return chi_sq(y_pred, arr_mu, arr_sig)


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

# In[Draw rand numbers]
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

# In[Tests]

def two_sample_test(mu1, mu2, sig_mu1, sig_mu2):
    """Compute p-value for mean values of two samples agreing with each other.
    Assumes Gaussian distribution."""
    z = (mu1-mu2)/np.sqrt(sig_mu1**2+sig_mu2**2)
    return std_to_prob(z) 

def runsTest(arr):
    """Runs test for randomness. 
    Parameters: arr array of digits
    Returns: p-value"""
    runs, n1, n2 = 0, 0, 0# Checking for start of new run
    median = np.median(arr)
    for i in range(len(arr)): 
        if (arr[i] >= median and arr[i-1] < median) or  (arr[i] < median and arr[i-1] >= median): 
                runs += 1  # no. of runs 
        if(arr[i]) >= median: 
            n1 += 1    # no. of positive values 
        else: 
            n2 += 1    # no. of negative values 
    runs_exp = ((2*n1*n2)/(n1+n2))+1
    stan_dev = np.sqrt((2*n1*n2*(2*n1*n2-n1-n2))/ 
                       (((n1+n2)**2)*(n1+n2-1))) 
  
    z = (runs-runs_exp)/stan_dev 
    p = std_to_prob(abs(z))
    return p 

def func_Poisson(x, N, lamb) :
    """Helper function for seq_freq_test."""
    if (x > -0.5) :
        return N * stats.poisson.pmf(x, lamb)
    else : 
        return 0.0

def seq_freq_test(integers, seq_l = 3, N_bins = 20):
    """Compare sequence frequency with Poisson hypothesis.
    Parameters:
        integers: array_like, input data
        seq_l: int,(default 3) length of sequence to be tested"""
    seq = []
    
    for i in range(-(seq_l-1), len(integers)-(seq_l-1) ) : 
        num = 0
        for j in range(seq_l):
            num+=integers[i+j]*int(10**(seq_l-1-j))
        seq.append(num)
    poisson_counts, _ = np.histogram(seq, int(10**(seq_l)+1), range=(-0.5, 10**seq_l+.5))
    xmin, xmax = -0.5, N_bins+.5
    
    fig, ax = plt.subplots(figsize=(12,8))
    hist_poisson = ax.hist(poisson_counts, N_bins, range=(xmin, xmax))
    
    counts, x_edges, _ = hist_poisson
    x_centers = 0.5*(x_edges[1:] + x_edges[:-1])
    x = x_centers[counts>0]
    y = counts[counts>0]
    sy = np.sqrt(y)

    chi2_object = Chi2Regression(func_Poisson, x, y, sy)
    minuit = Minuit(chi2_object, pedantic=False, N=10**seq_l, lamb=poisson_counts.mean())
    minuit.migrad()     # Launch the fit

    chi2_val = minuit.fval
    N_DOF = len(y) - len(minuit.args)
    chi2_prob = stats.chi2.sf(chi2_val, N_DOF)
    d = {'Entries'  : "{:d}".format(len(poisson_counts)),
     'Mean'     : "{:.5f}".format(poisson_counts.mean()),
     'STD Dev'  : "{:.5f}".format(poisson_counts.std(ddof=1)),
     'Chi2/ndf' : "{:.1f} / {:2d}".format(chi2_val, N_DOF),
     'Prob'     : "{:.6f}".format(chi2_prob),
     'N'        : "{:.1f} +/- {:.1f}".format(minuit.values['N'], minuit.errors['N']),
     'Lambda'   : "{:.2f} +/- {:.2f}".format(minuit.values['lamb'], minuit.errors['lamb'])
    }

    ax.text(0.55, 0.95, nice_string_output(d), family='monospace',
            transform=ax.transAxes, fontsize=14, verticalalignment='top');

    binwidth = (xmax-xmin) / N_bins 
    xaxis = np.linspace(xmin, xmax, 500)
    func_Poisson_vec = np.vectorize(func_Poisson)
    yaxis = binwidth*func_Poisson_vec(np.floor(xaxis+0.5), *minuit.args)
    ax.plot(xaxis, yaxis)
    
    return chi2_prob