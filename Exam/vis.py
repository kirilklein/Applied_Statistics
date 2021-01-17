# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 15:18:15 2021

@author: klein
"""
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Initializing Classifiers
from scipy import stats
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

def scatter_hist(X0,X1,Y0,Y1, ax, ax_histx, ax_histy, N_bins_x, N_bins_y):
    """Helper function to create additional axes with histograms."""
    
    #set tick parameters
    ax_histx.tick_params(axis="x", labelbottom=False,)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histx.tick_params(axis="y", labelsize = 18 )
    ax_histy.tick_params(axis ="x", labelsize = 18)
    
    # x ranges for histograms
    x_min =np.amin([X0,X1])
    x_max =np.amax([X0,X1])
    y_min =np.amin([Y0,Y1])
    y_max =np.amax([Y0,Y1])    
    
    # create actual histograms
    ax_histx.hist(X0, bins=N_bins_x,range=(x_min,x_max), alpha = .8)
    ax_histx.hist(X1, bins=N_bins_x,range=(x_min,x_max), alpha = .8)
    ax_histy.hist(Y0, bins=N_bins_y,range=(y_min,y_max),
                  orientation='horizontal', alpha = .8)
    ax_histy.hist(Y1, bins=N_bins_y, range=(y_min,y_max),
                  orientation='horizontal',alpha = .8)
    

def plot_classification(X, y, classifier, N_bins_x = 40, N_bins_y = 40, 
                        label0='H0', label1 = 'H1', xlabel='x', ylabel = 'y', 
                        figsize = (10,10), save = False,
                        figname = '', show = True):
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
    
    Returns:
    -------
    classifier: object, fitted classifier that contains 
        fit parameters and can be used to make predictions
    ax_scatter, ax_histx, ax_histy: axis objects, can be used to add a fit
        or change parameters
    fig: fig object, can be used for saving when sth changed
    """
    
    xx, yy = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max(), 100),
                     np.linspace(X[:,1].min(), X[:,1].max(), 100))    
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
                 N_bins_x=N_bins_x, N_bins_y=N_bins_y)
    if show:
        plt.show()
    if save:
        fig.savefig(figname, bbox_inches = 'tight')
    
    return classifier, ax_scatter, ax_histx, ax_histy, fig
    


_,_,ax_histx, ax_histy, fig = plot_classification(
        X,y, clf, save = True, figname = 'test2.png')
ax_histx.plot(np.linspace(-2,2,100), 10*stats.norm.pdf(np.linspace(-2,2,100), 
                                                      loc = 0,scale = .5))
display(fig)

