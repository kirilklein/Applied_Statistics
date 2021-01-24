# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 08:33:21 2020

@author: klein
"""
# In[Import]:
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
from scipy import stats
import sympy
import numpy as np
from IPython.core.display import Latex
from sympy import *
import custom_functions as cf
import importlib
import matplotlib as mpl
from matplotlib import gridspec
mpl.rcParams['figure.dpi'] = 100
importlib.reload(cf)
# In[Import custom]:
basepath = os.path.abspath('')
dpath = f"{basepath}/Pendulum"
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure

# In[Functions]
def g_an(L,T):  return L*(2*np.pi/T)**2
def minuit_fit(fit_function,x,y,ey=None):   
    Ndof_calc = len(x)-2
    chi2_object = Chi2Regression(fit_function, x, y, ey) 
    minuit = Minuit(chi2_object, pedantic=False, a=3, b=1, print_level=0)  
    minuit.migrad();  # perform the actual fit
    Chi2_fit = minuit.fval # the chi2 value
    Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_calc)
    return minuit, Chi2_fit, Prob_fit

def chi_sq(arr,sig_arr): return np.sum((arr-cf.weighted_avg(arr,sig_arr))**2/sig_arr**2)
def linear(x,a,b): return a*x+b

def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)
def gauss_extended(x, N, mu, sigma) :
    """Non-normalized Gaussian"""
    return N * gauss_pdf(x, mu, sigma)

def sig_L(D,sD,Lh,sLh, hm, shm):
    return np.sqrt((sD)**2+sLh**2+shm**2/20**2)

def sig_g(L,sL,T,sT): 
    return np.sqrt((2*np.pi/T)**4*sL**2+(8*L*np.pi**2/T**3*sT)**2)

# In[Compute length]
data_all = pd.read_csv("project_data.csv", skiprows = 24)

fig, ax = plt.subplots(3)
mu_hm = cf.df_arr(data_all.iloc[1:6,0])
s_hm = cf.df_arr(data_all.iloc[1:6,1])
wa_hm = cf.weighted_avg(mu_hm, s_hm)
swa_hm = cf.sig_weighted_avg(s_hm)
ax[0].hist(mu_hm)
chi2_hm = chi_sq(mu_hm,s_hm)
prob_hm = stats.chi2.pdf(chi_sq(mu_hm,s_hm), len(mu_hm))

mu_D  = cf.df_arr(data_all.iloc[1:6,2])
s_D = cf.df_arr(data_all.iloc[1:6,3])
wa_D = cf.weighted_avg(mu_D, s_D)
swa_D = cf.sig_weighted_avg(s_D)
ax[1].hist(mu_D)
chi2_D = chi_sq(mu_D,s_D)
prob_D = stats.chi2.pdf(chi2_D, len(mu_D))


mu_Lh = cf.df_arr(data_all.iloc[1:6,4])
s_Lh = cf.df_arr(data_all.iloc[1:6,5])
wa_Lh = cf.weighted_avg(mu_Lh, s_Lh)
swa_Lh = cf.sig_weighted_avg(s_Lh)
ax[2].hist(mu_Lh)
plt.subplots_adjust(hspace=.3)
chi2_Lh = chi_sq(mu_Lh,s_Lh)
prob_Lh = stats.chi2.pdf(chi2_Lh, len(mu_Lh))

mu_L = wa_D-wa_Lh+wa_hm/20
sL = sig_L(wa_D,swa_D,wa_Lh,swa_Lh,wa_hm,swa_hm)
# In[Compute times]
fn_times = [f for f in listdir(dpath) if isfile(join(dpath, f))]
Times = []
indices = [0,1,2,3,4,5]
fn_times_m = [fn_times[i] for i in indices]
for i in range(len(fn_times_m)):
    Times.append(np.genfromtxt(f'{dpath}/{fn_times_m[i]}', dtype = 'float'))
#Cut of last four values of neas data
Times[4] = Times[4][:-4]
#Cut of firs value of kiril data
Times[3] = Times[3][1:]
minuit_l = []
Chi2_l = []
Prob_l = []
T0 = []
for i in range(len(Times)):
    m,c,p = cf.minuit_fit(linear, Times[i][:,0], Times[i][:,1])
    minuit_l.append(m)
    Chi2_l.append(c)
    Prob_l.append(p)
    T0.append(m.values['a'])
T0 = np.array(T0)

# In[visualize]
fig, ax = plt.subplots(len(Times))
x = np.linspace(0,26,100)
Res, Res_std, Res_mean = [],[],[]
for i in range(len(Times)):
    x = Times[i][:,0]
    t = Times[i][:,1]
    y_fit = linear(x, *minuit_l[i].args)
    res = y_fit-t
    Res.append(res)
    Res_std.append(np.std(res))
    Res_mean.append(np.mean(np.abs(res)))
    ax[i].scatter(x, t)
    ax[i].plot(x, y_fit)
plt.subplots_adjust(hspace=.8)
# In[]
fig,ax = plt.subplots(len(Res),figsize = (13,15))
for i in range(len(Res)):
    ax[i].set_title(fn_times_m[i])
    ax[i].scatter(Times[i][:,0],Res[i])
plt.subplots_adjust(hspace=1)
# In[Test]
for i in range(6):
    fig,ax = plt.subplots()
    x =Times[i][:,0]
    ax.scatter(x, Times[i][:,1])
    ax.plot(x, linear(x, *minuit_l[i].args))
    ax.set_title(fn_times_m[i]+" "+str(i))
# In[Determine uncertainties]
Nbins = 10
minL = -.1
maxL = .1
binwidth = (maxL-minL)/Nbins
fig, ax = plt.subplots(len(Times))
for i in range(len(Times)):
    n,bins,_ = ax[i].hist(Res[i],bins=Nbins, range = (minL,maxL))

#remove 3rd and 4th entry

plt.subplots_adjust(hspace=.8)
binwidth = (maxL-minL)/Nbins
Unc_T_l = []
minuit_sT_l = []
mu_T_l = []
for i in range(len(Res)):
    ullh_T = UnbinnedLH(gauss_extended, Res[i], weights=None)
    minuit_T = Minuit(ullh_T, mu=Res[i].mean(),sigma=Res[i].std(ddof=1),
                      pedantic=False, N=len(Res[i])*binwidth, fix_N=True,
                     print_level=0) 
    minuit_T.migrad()
    minuit_sT_l.append(minuit_T)
    Unc_T_l.append(minuit_T.values['sigma'])
    mu_T_l.append(minuit_T.values['mu'])
# In[Res histogram]
fig, ax = plt.subplots(len(Times))
for i in range(len(Times)):
    xaxis = np.linspace(minL,maxL,300)
    yaxis = gauss_extended(xaxis, *minuit_sT_l[i].args) 
    ax[i].hist(Res[i],bins=Nbins, range = (minL,maxL))
    ax[i].plot(xaxis, yaxis)
    #looks goood
# In[Repeat fit but with unc]
minuit_l_f = []
Chi2_l_f = []
Prob_l_f = []
T_f, sT_f = [],[]
for i in range(len(Times)):
    m,c,p = cf.minuit_fit(linear, Times[i][:,0], 
                          Times[i][:,1], ey =Unc_T_l[i] )
    minuit_l_f.append(m)
    Chi2_l_f.append(c)
    Prob_l_f.append(p)
    T_f.append(m.values['a'])
    sT_f.append(m.errors['a'])
T_f = np.array(T_f)
sT_f = np.array(sT_f)
Chi2_T = chi_sq(T_f, sT_f)
Prob_T = stats.chi2.cdf(Chi2_T, len(T_f))
print(Prob_T,Chi2_T)
wa_T_f = cf.weighted_avg(T_f,sT_f)
swa_T_f = cf.sig_weighted_avg(sT_f)
gf = g_an((mu_L)/100,wa_T_f)
sgf = sig_g(mu_L/100,sL/100,wa_T_f,swa_T_f)
print(gf,sgf)

# In[Plots]
fig, ax = plt.subplots(2, figsize = (10,8),gridspec_kw={'height_ratios': [2, 1]})
                       
x_fit = np.linspace(0,26,100)
x = Times[1][:,0]
t = Times[1][:,1]


chi2_1 = minuit_l_f[1].fval
y_fit = linear(x_fit, *minuit_l_f[1].args)
offset = minuit_l_f[1].values['b']
doffset = minuit_l_f[1].errors['b']
text = "Fit results:\n"
text += "Period"+" = ({:4.4}".format(T_f[1])
text += r"$\pm$"+"{:3.1})".format(sT_f[1]) + "s\n"
text += "Offset"+" = ({:4.3}".format(offset)
text += r"$\pm$"+"{:3.1})".format(doffset) + "s\n"
text += "Chi2 = {:3.2}\n".format(Chi2_l_f[1])
text += "Prob(Chi2, ndof) = {:3.2}\n".format(Prob_l_f[1])
add_text_to_ax(0.02, 0.97, text, ax[0], fontsize=20)
ax[0].scatter(x, t)
ax[0].plot(x_fit, y_fit, 'r')
ax[0].set_ylabel('Time [s]',fontsize = 20)
ax[0].tick_params(axis = 'both',labelsize  = 19)
ax[0].set_xticks([])
ax[1].errorbar(x,Res[1],yerr = Unc_T_l[1],marker= 'o',
               linestyle = 'None')
ax[1].set_xlabel('Measurement',fontsize = 20)
ax[1].set_ylabel('Residual [s]', fontsize = 20)
ax[1].tick_params(axis = 'both',labelsize  = 19)
plt.subplots_adjust(hspace=0)