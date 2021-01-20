# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 15:18:15 2021

@author: klein
"""
from ExternalFunctions import UnbinnedLH, BinnedLH, Chi2Regression
from iminuit import Minuit
import numpy as np
import matplotlib.pyplot as plt
from ExternalFunctions import nice_string_output
from scipy.optimize import curve_fit
from scipy import stats
import as_toolbox

    
class Error(Exception):
    """Base class for other exceptions"""
    pass

class Invalid_Argument(Error):
    """Raised when an invalid argument was passed"""
    pass


def double_gauss_one_mean(x, N, f, mu, sig1, sig2):
    r"""Fit a double Gaussian with one mean
    Parameters:
        N: number of events
        f: fraction of events in 1st gauss
        sig1, sig2: widths
    Returns: N*(f*G(mu,sig1)+(1-f)*G(mu,sig2))"""
    return N*(f*stats.norm.pdf(x,mu,sig1)+(1-f)*stats.norm.pdf(x,mu,sig2))

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
    

def chi2_fit_mult_func(X,Y,SY,functions, P0, Ranges, show_plot = True, save_plot=False, figname = None, xlabel = 'x',ylabel='', 
                  data_label='Data, with Poisson errors',Fit_label = [''], Text_pos = [(0.01,.99), (.3,.99),(.4,.99)], 
                  figsize = (10,5), y_range= None, legend_loc = 2, legend_fs =18):
    r"""
    Fit piecewise defined functions, plot data with fit
    
    Parameters:
    ----------
    
    X, Y, SY: lists of array_like,
        input data, if produced from histogram, pass ONLY values where Y>0
    functions: list of callables,
        functions to fit to the data
    P0: list of array_like
        initial parameters for fit
    Ranges: list of (float,float),
        Ranges for piecewise defined functions
    Text_Pos: list of (float,float),
        Position to plot the resulting fit parameters
    Fit-label: list of str,
        Label for the fit functions displayed in the legend
    ...
    
    Returns:
    -------
    ax: axis object
    fig: figure object
    Popt: list of array_like fitted parameters
    Pcov: list of covariance matrices
    """
    
    fig, ax =plt.subplots(figsize= figsize)
    ax.errorbar(X, Y, yerr=SY, fmt='.b',  ecolor='b', elinewidth=.6, 
             capsize=0, capthick=0.1, label =data_label)#plot data
    color_cycle=['r', 'g', 'b', 'y']#colors for different fits
    Popt, Pcov = [], []#lists to store fitted params
    for i in range(len(functions)):
        func = functions[i]
        color = color_cycle[i]
        xmin, xmax = Ranges[i]
        mask = (xmin<X) & (xmax>=X)
        text_pos = Text_pos[i]
        fit_label = Fit_label[i]
        x, y, sy = X[mask], Y[mask], SY[mask]
        p0 = P0[i]
        popt, pcov = curve_fit(func, x,y, p0 = p0, sigma = sy)
        Popt.append(popt)
        Pcov.append(Pcov)
        sigma_popt = np.sqrt(np.diag(pcov))
        Ndof = len(x) - len(p0)
        chi2_val = as_toolbox.chi_sq(func(x, *popt), y, sy)
        Prob = stats.chi2.sf(chi2_val, Ndof)#p value
        xaxis = np.linspace(xmin, xmax, 1000)
        yaxis = func(xaxis, *popt)
        
        names = ['Chi2/NDF', 'Prob']
        values = [ "{:.2f} / {:d}".format(chi2_val, Ndof), "{:.3f}".format(Prob),]
        for i in range(len(p0)):
            varname = func.__code__.co_varnames[1+i]
            names.append(varname)
            values.append("{:.3f} +/- {:.3f}".format(popt[i], sigma_popt[i]))
        
        d={}
        for n,v in zip(names,values):
            d[n]=v
    
        text = nice_string_output(d, extra_spacing=0, decimals=2)
        ax.text(text_pos[0], text_pos[1], text, fontsize=14,  family='monospace', 
            transform=ax.transAxes, color=color, verticalalignment='top', horizontalalignment ='left');
        ax.plot(xaxis, yaxis,color = color, label= fit_label);
    ax.set_ylabel(ylabel, fontsize = 25)
    ax.tick_params(axis = 'both',labelsize  = 20)
    ax.legend(loc = legend_loc, fontsize = legend_fs)
    ax.set_xlabel(xlabel, fontsize =20)
    ax.tick_params(labelsize =20)
    if y_range!=None:
        ax.set_ylim((y_range[0],y_range[1]))
    fig.tight_layout()

    if save_plot:
        fig.savefig(figname)
        
    if show_plot:
        plt.show(fig)
    else:
        plt.close(fig)
    return ax, fig, Popt, Pcov
        