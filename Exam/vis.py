# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 15:18:15 2021

@author: klein
"""
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from mlxtend.plotting import plot_decision_regions
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Initializing Classifiers
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
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False,)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histx.tick_params(axis="y", labelsize = 18 )
    ax_histy.tick_params(axis ="x", labelsize = 18)
    
    # now determine nice limits by hand:
    x_min =np.amin([X0,X1])
    x_max =np.amax([X0,X1])
    y_min =np.amin([Y0,Y1])
    y_max =np.amax([Y0,Y1])    
    
    ax_histx.hist(X0, bins=N_bins_x,range=(x_min,x_max))
    ax_histx.hist(X1, bins=N_bins_x,range=(x_min,x_max))
    ax_histy.hist(Y0, bins=N_bins_y,range=(y_min,y_max),
                  orientation='horizontal')
    ax_histy.hist(Y1, bins=N_bins_y, range=(y_min,y_max),
                  orientation='horizontal')
    

def plot_classification(X, y, classifier, label0='H0', label1 = 'H1'):
    xx, yy = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max(), 50),
                     np.linspace(X[:,1].min(), X[:,1].max(), 50), 
                     xlabel='x', ylabel = 'y')
    
    classifier.fit(X, y)
    fig, ax = plt.subplots()
    
    scatter_kwargs = {'s': 50, 'edgecolor': 'k', 'alpha': 1,
                      'marker':'s'}
    contourf_kwargs = {'alpha': .4}

    ax_dr = plot_decision_regions(X=X, y=y, 
                                clf=classifier, legend=2,
                                scatter_kwargs=scatter_kwargs,
                                contourf_kwargs=contourf_kwargs,#
                                ax = ax)
    
    #fig = plt.figure(figsize=(10,8))
    #ax_dr.set_title('Linear Discriminant Analysis', fontsize= 20)
    ax_dr.tick_params(labelsize = 20)

    # no labels
    handles, labels = ax_dr.get_legend_handles_labels()
    
    bbox = ax_dr.get_position()
    
    ##########################################
    # definitions for the axes
    left, bottom, width, height = bbox.bounds
    spacing = 0.03
    ax_dr.legend(handles, [label0, label1], 
           framealpha=0.3, scatterpoints=1, fontsize = 20,
            bbox_to_anchor=(left+width, bottom+height, 0.5, 0.5))

    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]


    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    mask0 = y==0
    X0 = X[mask0,0]
    X1 = X[~mask0,0]
    Y0 = X[mask0,1]
    Y1 = X[~mask0,1]
    scatter_hist(X0,X1,Y0, Y1, ax_dr, ax_histx, ax_histy,
                 N_bins_x=40, N_bins_y=40)
    ########################
    
    
    plt.show()
    return classifier
    
plot_classification(X,y, clf)
