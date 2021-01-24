# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 15:18:15 2021

@author: klein
"""
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from scipy import stats
import fits
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
##################################################################################
# In[General plots]
def nice_histogram(x_all, N_bins, show_plot = False, plot_hist = False, plot_errors = True, 
                   plot_legend = True, save = False, figname = '', x_range = None, 
                   data_label = 'Data, histogram', data_label_hist = '',figsize = (12,6), 
                   histtype = 'step', color_hist = 'orange', fmt = '.b', ecolor = 'b',
                   xlabel = 'x', ylabel = 'Frequency', label_fs = 20, 
                   legend_fs = 18, legend_loc = 0, ticks_lsize = 20, xlog_scale = False,
                   ylog_scale = False, axis = None, figure = None, dpi = 80):
    """Produce a nice histogram.
    Returns: x, y, sy, binwidth, fig, ax."""
    
    if not(x_range==None):
        mask_x = (x_all>x_range[0]) & (x_all<x_range[1])
        x_all = x_all[mask_x]
    x,y,sy, binwidth = fits.produce_hist_values(x_all,N_bins, x_range = x_range,log = xlog_scale)
    if not(axis==None):
        ax = axis
        fig = figure
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
    if plot_hist:
        ax.hist(x_all, bins=N_bins, range=(x_all.min(), x_all.max()), histtype=histtype,
                       linewidth=2, color=color_hist, label=data_label_hist)
    if plot_errors:
        ax.errorbar(x, y, yerr=sy, xerr=0.0, label=data_label, fmt=fmt,
                    ecolor=ecolor, elinewidth=1, capsize=1, capthick=1)
    ax.set_xlabel(xlabel, fontsize = label_fs)
    ax.set_ylabel(ylabel, fontsize = label_fs)
    ax.tick_params(axis ='both', labelsize = ticks_lsize)
    if xlog_scale:
        ax.set_xscale('log')
    if ylog_scale:
        ax.set_yscale('log')
    if plot_legend:
        ax.legend(loc=legend_loc, fontsize = legend_fs)
        
    
    if save:
        fig.tight_layout()
        fig.savefig(figname, dpi = dpi)
    
    if show_plot:
        plt.show(fig)
    else:
        plt.close(fig)
    
    return x, y, sy, binwidth, fig, ax


def scatter_hist(X0,X1,Y0,Y1, ax, ax_histx, ax_histy, N_bins_x, N_bins_y,
                 histlabel0, histlabel1):
    """Helper function to create additional axes with histograms."""
    
    #set tick parameters
    ax_histx.tick_params(axis="x", labelbottom=False,)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histx.tick_params(axis="y", labelsize = 18 )
    ax_histy.tick_params(axis ="x", labelsize = 18)
    
    # x ranges for histograms
    x_min = np.amin([X0,X1])
    x_max = np.amax([X0,X1])
    y_min = np.amin([Y0,Y1])
    y_max = np.amax([Y0,Y1])    
    
    # create actual histograms
    ax_histx.hist(X0, bins=N_bins_x,range=(x_min,x_max), 
                  alpha = .8, label = histlabel0)
    ax_histx.hist(X1, bins=N_bins_x,range=(x_min,x_max), 
                  alpha = .8, label = histlabel1)
    ax_histy.hist(Y0, bins=N_bins_y,range=(y_min,y_max),
                  orientation='horizontal', alpha = .8)
    ax_histy.hist(Y1, bins=N_bins_y, range=(y_min,y_max),
                  orientation='horizontal',alpha = .8)
    ax_histx.legend(fontsize = 12)
#############################################################################

def plot_classification(X, y, classifier, N_bins_x = 40, N_bins_y = 40, 
                        label0='type I', label1 = 'type II', 
                        histlabel0 = 'type I', histlabel1 = 'type II',
                        xlabel='x', ylabel = 'y', 
                        figsize = (10,10), save = False, figname = '',
                        show_plot = False):

    r"""
    Create a scatter plot including separation according to classifier #
    with histograms of projections on x and y axis.
    
    Parameters:
    -----------
    X: array_like
        Input data with N rows and 2 columns, where N is the number of samples
        and 2 variables
    y: array_like
        Target data with N rows and 1 column, contains 0s and 1s
    classifier: class from sklearn
    label0, label1: str, optional,
        labels for first and second variables, default H0,H1
    xlabel, ylabel: str, optional, default x, y
    save: bool, default(False)
    figname: if save==True 
        saves figure under figname
    N_bins_x, N_bins_y: int 
        specifies number of bins for first and second variable
    label0, label1, histlabel0, histlabel1: str, optional
    xlabel, ylabel: str, optional
    figsize: (float,float), optional, default (10,10)
    show: bool, show plot if True
    
    Returns:
    -------
    classifier: object, fitted classifier that contains 
        fit parameters and can be used to make predictions
    ax_scatter, ax_histx, ax_histy: axis objects, can be used to add a fit
        or change parameters
    fig: fig object, can be used for saving when sth changed
    """
    
    
    classifier.fit(X, y)
    
    fig, ax = plt.subplots()#create figure
    scatter_kwargs = {'s': 100, 'edgecolor': 'k', 'alpha': 1,
                      'marker':'s'}
    contourf_kwargs = {'alpha': .3}
    #plot decision boundaries
    ax_scatter = plot_decision_regions(X=X, y=y, 
                                clf=classifier, legend=2,
                                scatter_kwargs=scatter_kwargs,
                                contourf_kwargs=contourf_kwargs,#
                                ax = ax)
    ax_scatter.set_xlim(X[:,0].min()*1.1, X[:,0].max()*1.1)
    ax_scatter.set_ylim(X[:,1].min()*1.1, X[:,1].max()*1.1)
    ax_scatter.tick_params(labelsize = 20)
    ax_scatter.set_xlabel(xlabel, fontsize = 20)
    ax_scatter.set_ylabel(ylabel, fontsize = 20)
    handles, labels = ax_scatter.get_legend_handles_labels()#get pos of axis  
    bbox = ax_scatter.get_position()
    left, bottom, width, height = bbox.bounds
    spacing = 0.04
    ax_scatter.legend(handles, [label0, label1], 
           framealpha=0.3, scatterpoints=1, fontsize = 20,
            bbox_to_anchor=(left+width, bottom+height, 0.5, 0.5))
    #set position for additional axes
    rect_histx = [left, bottom + height + spacing, width, 0.25]
    rect_histy = [left + width + spacing, bottom, 0.25, height]


    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    mask0 = y==0
    X0 = X[mask0,0]
    X1 = X[~mask0,0]
    Y0 = X[mask0,1]
    Y1 = X[~mask0,1]
    scatter_hist(X0,X1,Y0, Y1, ax_scatter, ax_histx, ax_histy,
                 N_bins_x=N_bins_x, N_bins_y=N_bins_y, 
                 histlabel0 = histlabel0 , histlabel1 = histlabel1)
    
    if save:
        fig.savefig(figname, bbox_inches = 'tight')
    
    if show_plot:
        plt.show(fig)
    else:
        plt.close(fig)
    
    return classifier, ax_scatter, ax_histx, ax_histy, fig
############################################################################# 

def add_zoom_inset(ax, zoom,loc, x,y, xlim, ylim , sy = None, 
                   xlabel = '', ylabel = '', label_fs = 18,
                   mark_inset_loc = (3,1), borderpad = 4):
    """Add inset axis that shows a region of the data.
    
    Parameters:
    -----------
    ax: axis object
    zoom: float, zoom factor
    loc: integer, location of inset axis
    x,y, sy: array_like, data to plot
    xlim, ylim: (float, float) limits for x and y axis
    xlabel, ylabel: str, label for x and y axis
    label_fs: float, fonstsize for x and y axis labels
    mark_inset_loc: (int, int), corners for connection to the new axis
    borderpad: float, distance from border
    """
    
    axins = zoomed_inset_axes(ax,zoom,loc = loc,
                              borderpad=borderpad) 
    if sy==None:
        axins.plot(x, y)
    else: 
        axins.errorbar(x, y, yerr=sy, fmt='.b',  ecolor='b', elinewidth=.5, 
             capsize=0, capthick=0.1)
        
    if not(ylim==None):
        axins.set_ylim(ylim)
    
    axins.set_xlabel(xlabel, fontsize = label_fs)
    axins.set_ylabel(ylabel, fontsize = label_fs)
    axins.set_xlim(xlim) # Limit the region for zoom
    mark_inset(ax, axins, loc1=mark_inset_loc[0], loc2=mark_inset_loc[1], fc="none", ec="0.5")
   

# In[Random Numbers] 

def create_1d_hist(ax, values, bins, x_range, title):
    """Helper function for show_int_distribution. (Author: Troels Petersen)"""
    ax.hist(values, bins, x_range, histtype='step', density=False, lw=2)         
    ax.set(xlim=x_range, title=title)
    hist_data = np.histogram(values, bins, x_range)
    return hist_data

def get_chi2_ndf( hist, const):
    """Helper function for show_int_distribution. (Author: Troels Petersen)"""
    data = hist[0]
    const_unnormed = const * data.sum()
    chi2 = np.sum( (data - const_unnormed)**2 / data )
    ndof = data.size
    return chi2, ndof

def show_int_distribution(integers, save_plot = True, figname = '', show_plot= False):
    """Show histogram of integers, to see if random.(Author: Troels Petersen)
    modified by: Kiril Klein
    Parameters: 
        integers, array_like
    Returns: 
        dict_raw, dict_odd_even, dict_high_low: dictionaries
            contains chi2, ndf, p for the hypothesis of integers being random.
        AX: list of axes objects
        fig: figure
        """
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    ax_number, ax_odd_even, ax_high_low = ax.flatten()

    # Fill 1d histograms and plot them:
    hist_numbers  = create_1d_hist(ax_number,   integers,     10, (-0.5, 9.5), 'Numbers posted')                # Plot all digits
    hist_odd_even = create_1d_hist(ax_odd_even, integers % 2,  2, (-0.5, 1.5), 'Even and odd numbers')          # Is number even or odd
    hist_high_low = create_1d_hist(ax_high_low, integers // 5, 2, (-0.5, 1.5), 'Above and equal to or below 5') # Is number >= or < 5
    fig.tight_layout()
    
    chi2_raw, ndf_raw  = get_chi2_ndf( hist_numbers,  1.0 / 10)
    chi2_odd_even, ndf_odd_even = get_chi2_ndf( hist_odd_even, 1.0 / 2 )
    chi2_high_low, ndf_high_low = get_chi2_ndf( hist_high_low, 1.0 / 2 )
    p_raw = stats.chi2.sf(chi2_raw, ndf_raw)
    p_odd_even = stats.chi2.sf(chi2_odd_even, ndf_odd_even)
    p_high_low = stats.chi2.sf(chi2_high_low, ndf_high_low)
    
    dict_raw = {'chi2':chi2_raw, 'ndf':ndf_raw ,'p': p_raw }
    dict_odd_even = {'chi2':chi2_odd_even, 'ndf':ndf_odd_even ,'p': p_odd_even }
    dict_high_low = {'chi2':chi2_high_low, 'ndf':ndf_high_low ,'p': p_high_low }
    AX = [ax_number, ax_odd_even, ax_high_low]
    if save_plot:
        fig.savefig(figname)
    if show_plot:
        plt.show(fig)
    else:
        plt.close(fig)
    
    return dict_raw, dict_odd_even, dict_high_low, fig, AX

# In[Example]
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from sklearn.metrics import plot_roc_curve

clf = LinearDiscriminantAnalysis(solver='svd')

rng = np.random.RandomState(0)
x1v1 = np.random.normal(-1,.2,200)
x2v1 = np.random.normal(0,.5,200)
x1v2 = np.random.normal(-1,.2,200)
x2v2 = np.random.normal(0,.5,200)

y1 = np.zeros(200, dtype = 'int')
y2 = np.ones(200, dtype = 'int')
X1 = np.concatenate((x1v1[:,np.newaxis],x1v2[:,np.newaxis]), axis = 1)
X2 = np.concatenate((x2v1[:,np.newaxis],x2v2[:,np.newaxis]), axis = 1)
X = np.concatenate((X1,X2))
y = np.concatenate((y1,y2))
fitted_clf,_,ax_histx, ax_histy, fig = plot_classification(
        X,y, clf, save = True, figname = 'test2.png', show = False)
ax_histx.plot(np.linspace(-2,2,100), 10*stats.norm.pdf(np.linspace(-2,2,100), 
                                                      loc = 0,scale = .5))
display(fig)
plot_roc_curve(fitted_clf, X, y)
"""