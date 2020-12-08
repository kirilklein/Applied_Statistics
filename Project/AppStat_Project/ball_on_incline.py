#!/usr/bin/env python
# coding: utf-8

# In[198]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import seaborn as sns
import sys
from iminuit import Minuit
from scipy import signal

# In[199]:
basepath = os.path.abspath('')
workpath = os.path.dirname(basepath)
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure

# In[200]:
def weighted_avg(mu,sig):
    """Computes weighted average given mean values and sigma"""
    return np.sum(mu/sig**2)/np.sum(1/sig**2)
def df_arr(data): return np.array(data,dtype='float32')
def quadr(x,a,b,c): return a*x**2+b*x+c
def fast_hist(data, bins ):
    fig, ax = plt.subplots()
    ax.hist(data, bins = bins)
    plt.show()
    
def step_func(x,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,c1,c2):
    if x<a1:
        y = c1
    elif (x>a1) & (x<a2):
        y = c2
    elif (x>a2) & (x<a3):
        y = c1
    elif (x>a3) & (x<a4):
        y = c2
    elif (x>a4) & (x<a5):
        y = c1
    elif (x>a5) & (x<a6):
        y = c2
    elif (x>a6) & (x<a7):
        y = c1
    elif (x>a7) & (x<a8):
        y = c2
    elif (x>a8) & (x<a9):
        y = c1
    elif (x>a9) & (x<a10):
        y = c2
    else:
        y = c1
    return y
def fit_func(data):
    x,y = np.array(data[0]['Time (s)']),np.array(data[0]['Channel 1 (V)'])
    chi2_object = Chi2Regression(step_func,x, y)
    minuit = Minuit(chi2_object, pedantic=False,a1=-.8,a2=-.75 ,a3=-.6,a4=-.55,a5=-.48,
                           a6=-.4,a7=-.35,a8=-.3,a9=-.2,c1=0.,c2=2.5 )
    minuit.migrad();
    return minuit.args

def determine_times(data,index):
    x, y = x,y = np.array(data[index]['Time (s)']),np.array(data[index]['Channel 1 (V)'])
    mask0 = y>1 #separate in intervals of peaks and background
    x0 = x[mask0]
    y0 = y[mask0]
    dx0 = np.roll(x0,-1)-x0
    return dx0

# In[202]:


data = Time1b
index  = 1
x, y = np.array(data[index]['Time (s)']),np.array(data[index]['Channel 1 (V)'])
ind, params = signal.find_peaks(y,height = (2,3),width = (1000,2900))
x_v = x[ind]
widths = params['widths']
ind_l = (ind-widths/2).astype(int)
ind_r = (ind+widths/2).astype(int)
x_l = x[ind_l]
x_r = x[ind_r]
fig,ax = plt.subplots()
ax.plot(x,y)
#ax.set_xlim(-.,-.7)
ax.vlines(x_v, ymin = 0, ymax = 3, colors = 'k')
ax.vlines(x_l,ymin = 0, ymax = 3,colors = 'r')
ax.vlines(x_r,ymin = 0, ymax = 3,colors = 'b')


# In[208]:


path_m1b = f"{basepath}/meas1/big"
path_m1s = f"{basepath}/meas1/small"
path_m2b = f"{basepath}/meas2/big"
path_m2s = f"{basepath}/meas2/small"
# In[219]:


Time1s[0].shape
fn_m1b


# In[209]:


fn_m1b = [f for f in listdir(path_m1b) if isfile(join(path_m1b, f))]
fn_m1s = [f for f in listdir(path_m1s) if isfile(join(path_m1s, f))]
fn_m2b = [f for f in listdir(path_m2b) if isfile(join(path_m2b, f))]
fn_m2s = [f for f in listdir(path_m2s) if isfile(join(path_m2s, f))]
Time1b, Time1s, Time2b, Time2s = [],[],[],[]
Time1bp, Time1sp, Time2bp, Time2sp = [],[],[],[]
for i in range(len(fn_m1b)):
    Time1b.append(pd.read_csv(f'{path_m1b}/{fn_m1b[i]}',skiprows = 14 ))
for i in range(len(fn_m1s)):
    Time1s.append(pd.read_csv(f'{path_m1s}/{fn_m1s[i]}',skiprows = 14 ))
for i in range(len(fn_m2b)):
    Time2b.append(pd.read_csv(f'{path_m2b}/{fn_m2b[i]}',skiprows = 14 ))
for i in range(len(fn_m2s)):
    Time2s.append(pd.read_csv(f'{path_m2s}/{fn_m2s[i]}',skiprows = 14 ))


# In[249]:


pd.read_csv(f'{path_m2b}/{fn_m2b[0]}',skiprows = 14 )


# In[247]:


Time2b[0], fn_m2b[0]


# In[217]:


xlab, ylab = 'Time (s)', 'Channel 1 (V)' 
files = fn_m1s
fig, ax = plt.subplots(len(files))
for i in range(len(files)):
    ax[i].plot(Time1s[i][xlab], Time1s[i][ylab])
    ax[i].set_xlim((-.9,0.5))
plt.show()


# In[211]:


data = pd.read_csv('project_data.csv', sep = ',',skiprows = 2)
data.head(16)


# ## Separate data

# In[212]:


mu_angle = df_arr(data.iloc[:15,0])
s_angle = df_arr(data.iloc[:15,1])
wa_angle = weighted_avg(mu_angle, s_angle)
mu_Db = df_arr(data['D_big'][1:5])
mu_Ds = df_arr(data['D_small'][1:5])
s_D = 0.05*np.ones(4)
mu_drail = df_arr(data['d_rail'][:4])
s_drail = df_arr(data['d_rail unc'][:4])
mu_angle_r = df_arr(data['angle rotated'][:15])
s_angle_r = df_arr(data['angle rot. unc'][:15])
P_all = []
for i in range(5):
    P_all.append(data.iloc[:4,2+i])
P_all[0]

