#!/usr/bin/env python
# coding: utf-8

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

# In[Import custom]:
basepath = os.path.abspath('')
dpath = f"{basepath}/Incline"
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure

# In[Define functions]:
def weighted_avg(mu,sig):
    """Computes weighted average given mean values and sigma"""
    return np.sum(mu/sig**2)/np.sum(1/sig**2)

def sig_weighted_avg(sig):
    return 1/np.sqrt(np.sum(1/sig**2))

def df_arr(data): return np.array(data,dtype='float32')

def quadr(x,a,b,c): return 1/2*a*x**2+b*x+c

def fast_hist(data, bins ):
    fig, ax = plt.subplots()
    ax.hist(data, bins = bins)
    plt.show()
    

def data_to_time(data,index):
    """returns left boundary, 
    centre and right boundary of the peaks"""
    x, y = np.array(data[index]['Time (s)']),np.array(data[index]['Channel 1 (V)'])
    ind, params = signal.find_peaks(y,height = (2,3),width = (1000,3000))
    x_c = x[ind]
    widths = params['widths']
    ind_l = (ind-widths/2).astype(int)
    ind_r = (ind+widths/2).astype(int)
    x_l = x[ind_l]
    x_r = x[ind_r]
    return x_l, x_c, x_r

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
    
def deg_rad(d): return 2*np.pi*d/360
    
def chi2(data, mu):
    return np.sum((data-mu)**2/mu)
def minuit_fit(fit_function,x,y,ey):   
    Ndof_calc = len(x)-3
    chi2_object = Chi2Regression(fit_function, x, y, ey) 
    minuit = Minuit(chi2_object, pedantic=False, a=70, b=100,c=50, print_level=0)  
    minuit.migrad();  # perform the actual fit
    Chi2_fit = minuit.fval # the chi2 value
    Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_calc)
    return minuit, Chi2_fit, Prob_fit
# In[Error prop]

# Define variables:
g,a,theta,dtheta,D,d = symbols("g,a,theta,dtheta,D,d")
sa,stheta,sdtheta,sD,sd,sg = symbols("sigma_a, sigma_theta, sigma_dtheta, sigma_D, sigma_d, sigma_g")

# Perimeter:
# Define relation, and print:
g = a/sin(theta+dtheta)*(1+2/5*D**2/(D**2-d**2))
lprint(latex(Eq(symbols('g'),g)))

# Calculate uncertainty and print:
sg = sqrt((g.diff(a) * sa)**2 + (g.diff(theta) * stheta)**2 + 
          (g.diff(dtheta) * sdtheta)**2+ (g.diff(D) * sD)**2+
          (g.diff(d) * sd)**2)
lprint(latex(Eq(symbols('sigma_g'), sg)))

# Turn expression into numerical functions 
fg = sympy.lambdify((a,theta,dtheta,D,d),g, ("math", "mpmath", "sympy"))
fsg = sympy.lambdify((a,sa,theta,stheta,dtheta,sdtheta,D,sD,d,sd),sg,("math", "mpmath", "sympy"))

# Define values and their errors
va, vsa = 2,.1
vtheta, vstheta = deg_rad(30), deg_rad(1) 
vdtheta,vsdtheta = deg_rad(1), deg_rad(.5)
vD, vsD = .03, .001
vd, vsd = .01, .001
# Numerically evaluate expressions and print 
vg = fg(va,vtheta,vdtheta,vD,vd)
vsg = fsg(va,vsa,vtheta,vstheta,vdtheta,vsdtheta,vD,vsD,vd,vsd)
print(vg)
print(vsg)
# In[Specify paths:
path_m1b = f"{dpath}/meas1b"
path_m1s = f"{dpath}/meas1s"
path_m2b = f"{dpath}/meas2b"
path_m2s = f"{dpath}/meas2s"

# In[Load times]:
fn_m1b = [f for f in listdir(path_m1b) if isfile(join(path_m1b, f))]
fn_m1s = [f for f in listdir(path_m1s) if isfile(join(path_m1s, f))]
fn_m2b = [f for f in listdir(path_m2b) if isfile(join(path_m2b, f))]
fn_m2s = [f for f in listdir(path_m2s) if isfile(join(path_m2s, f))]
Time1b, Time1s, Time2b, Time2s = [],[],[],[]
tl_1b, tl_1s, tl_2b, tl_2s = [],[],[],[]
for i in range(len(fn_m1b)):
    Time1b.append(pd.read_csv(f'{path_m1b}/{fn_m1b[i]}',skiprows = 14 ))
for i in range(len(fn_m1s)):
    Time1s.append(pd.read_csv(f'{path_m1s}/{fn_m1s[i]}',skiprows = 14 ))
for i in range(len(fn_m2b)):
    Time2b.append(pd.read_csv(f'{path_m2b}/{fn_m2b[i]}',skiprows = 14 ))
for i in range(len(fn_m2s)):
    Time2s.append(pd.read_csv(f'{path_m2s}/{fn_m2s[i]}',skiprows = 14 ))
for i in range(len(Time1b)):
    tl,_,_ = data_to_time(Time1b,i)
    tl_1b.append(tl)
for i in range(len(Time1s)):
    tl,_,_ = data_to_time(Time1s,i)
    tl_1s.append(tl)
for i in range(len(Time2b)):
    tl,_,_ = data_to_time(Time2b,i)
    tl_2b.append(tl)
for i in range(len(Time2s)):
    tl,_,_ = data_to_time(Time2s,i)
    tl_2s.append(tl)
tl_1b, tl_1s  =  np.array(tl_1b), np.array(tl_1s)
tl_2b, tl_2s = np.array(tl_2b), np.array(tl_2s)
# In[just visualization]:

xlab, ylab = 'Time (s)', 'Channel 1 (V)' 
files = fn_m1b
data = Time1b
times = tl_1b

fig, ax = plt.subplots(len(files))
for i in range(len(files)):
    ax[i].plot(data[i][xlab], data[i][ylab])
    ax[i].vlines(times[i],ymin = 0, ymax = 3,colors = 'r')
    #ax[i].set_xlim((-.8,.7))
plt.show()

# In[Store data in array]
data_all = pd.read_csv("project_data.csv", skiprows = 2)

mu_angle = df_arr(data_all.iloc[:15,0])
s_angle = df_arr(data_all.iloc[:15,1])
wa_angle = weighted_avg(mu_angle, s_angle)
swa_angle = sig_weighted_avg(s_angle)


mu_Db = df_arr(data_all['D_big'][1:5])
mu_Ds = df_arr(data_all['D_small'][1:5])
s_D = 0.05*np.ones(len(mu_Db))
wa_Db = weighted_avg(mu_Db, s_D)
wa_Ds = weighted_avg(mu_Ds, s_D)
swa_Ds = sig_weighted_avg(s_D)
swa_Db = swa_Ds

mu_drail = df_arr(data_all['d_rail'][:4])
s_drail = df_arr(data_all['d_rail unc'][:4])
wa_drail = weighted_avg(mu_drail, s_drail)
swa_drail = sig_weighted_avg(s_drail)

mu_angle_r = df_arr(data_all['angle rotated'][:15])
s_angle_r = df_arr(data_all['angle rot. unc'][:15])
wa_angle_r = weighted_avg(mu_angle_r, s_angle_r)
swa_angle_r = sig_weighted_avg(s_angle_r)

P_all = []
for i in range(5):
    P_all.append(data_all.iloc[:4,2+i])
P_all = np.array(P_all)
s_P = .05*np.ones(P_all.shape[1])
wa_P, swa_P = [], []
P_all = P_all.astype(dtype = 'float32')
for i in range(5):
    wa_P.append(weighted_avg(P_all[i],s_P))
    swa_P.append(sig_weighted_avg(s_P))
wa_P = np.array(wa_P)
swa_P = np.array(swa_P)

mu_a = df_arr(data_all['a'][1:4])
mu_b = df_arr(data_all['b'][1:4])
mu_c = df_arr(data_all['c'][1:4])
s_abc = np.ones(4)*.05
diff = np.sqrt(mu_a**2+mu_b**2)-mu_c
print(diff)
print(np.tan(mu_b/mu_a)*360/(2*np.pi))
# In[Determine acceleration]

a_1b, sa_1b, args_1b = [],[], []
a_1s, sa_1s, args_1s = [],[], []
a_2b, sa_2b, args_2b = [],[], []
a_2s, sa_2s, args_2s = [],[], []
for i in range(len(tl_1b)):
    minuit_obj,Chi2, Prob_1b = minuit_fit(quadr, tl_1b[i],wa_P, swa_P )
    a = minuit_obj.values['a']
    sa = minuit_obj.errors['a']
    a_1b.append(a)
    sa_1b.append(sa)
    args_1b.append(minuit_obj.args)
for i in range(len(tl_1s)):
    minuit_obj,Chi2, Prob1s = minuit_fit(quadr, tl_1s[i],wa_P, swa_P )
    a = minuit_obj.values['a']
    sa = minuit_obj.errors['a']
    a_1s.append(a)
    sa_1s.append(sa)
    args_1s.append(minuit_obj.args)
for i in range(len(tl_2b)):
    minuit_obj,Chi2, Prob2b = minuit_fit(quadr, tl_2b[i],wa_P, swa_P )
    a = minuit_obj.values['a']
    sa = minuit_obj.errors['a']
    a_2b.append(a)
    sa_2b.append(sa)
    args_2b.append(minuit_obj.args)
for i in range(len(tl_2s)):
    minuit_obj,Chi2, Prob2s = minuit_fit(quadr, tl_2s[i],wa_P, swa_P )
    a = minuit_obj.values['a']
    sa = minuit_obj.errors['a']
    a_2s.append(a)
    sa_2s.append(sa)
    args_2s.append(minuit_obj.args)    
a_1b,a_1s,a_2b,a_2s = np.array(a_1b), np.array(a_1s), np.array(a_2b), np.array(a_2s)
sa_1b,sa_1s,sa_2b,sa_2s = np.array(sa_1b), np.array(sa_1s), np.array(sa_2b), np.array(sa_2s)
wa_a_1b = weighted_avg(a_1b,sa_1b)
wa_a_1s = weighted_avg(a_1s,sa_1s)
wa_a_2b = weighted_avg(a_2b,sa_2b)
wa_a_2s = weighted_avg(a_2s,sa_2s)


fig,ax = plt.subplots()
ax.plot(tl_2s.T,wa_P,'.')
x = np.linspace(-.8,0.4,100)
for i in range(len(tl_2s)):
    ax.plot(x, quadr(x,*args_2s[i]))
print(a/100,sa/100)

 
# In[]

def g_val(a0, theta0, dtheta0, D0, d0):
    return a0/np.sin(theta0+dtheta0)*(1+2/5*D0**2/(D0**2-d0**2))
#gv = g_val(wa_a_1b/100,deg_rad(wa_angle),0,wa_Db/1e3,wa_drail/1e3)
g_test = g_val(1.46, 13*2*np.pi/360,0, 14./1e3, 5.81/1e3)
print(g_test)
# In[Compute g]
wa_a_1b = weighted_avg(a_1b,sa_1b)

va, vsa = wa_a_1b/100, np.sqrt(1/np.sum(1/sa_1b**2))/100
vtheta, vstheta = deg_rad(wa_angle), 2*np.pi*deg_rad(swa_angle)/360 
vdtheta,vsdtheta = 0,0#deg_rad(1), deg_rad(.5)
vD, svD = wa_Db/1e3, swa_Db/1e3
vd, svd = wa_drail/1e3, swa_drail/1e3

gv =  g_val(wa_a_1b/100,deg_rad(wa_angle),0,wa_Db/1e3,wa_drail/1e3)

g = fg(va,vtheta,vdtheta,vD,vd)
print(fg(va,vtheta,vdtheta,vD,vd))
sg = fsg(va,vsa,vtheta,vstheta,vdtheta,vsdtheta,vD,vsD,vd,vsd)
# In[]

wa_a_1s = weighted_avg(a_1s,sa_1s)

va, vsa = wa_a_1s/100, np.sqrt(1/np.sum(1/sa_1s**2))/100
vtheta, vstheta = deg_rad(wa_angle), 2*np.pi*deg_rad(swa_angle)/360 
vdtheta,vsdtheta = 0,0#deg_rad(1), deg_rad(.5)

vD, svD = wa_Ds/1e3, swa_Ds/1e3
vd, svd = wa_drail/1e3, swa_drail/1e3

gv =  g_val(wa_a_1s/100,deg_rad(wa_angle),0,wa_Ds/1e3,wa_drail/1e3)
g = fg(va,vtheta,vdtheta,vD,vd)
print(fg(va,vtheta,vdtheta,vD,vd))
sg = fsg(va,vsa,vtheta,vstheta,vdtheta,vsdtheta,vD,vsD,vd,vsd)

# In[]

wa_a_2s = weighted_avg(a_2s,sa_2s)

va, vsa = wa_a_2s/100, np.sqrt(1/np.sum(1/sa_2s**2))/100
vtheta, vstheta = deg_rad(wa_angle_r), 2*np.pi*deg_rad(swa_angle_r)/360 
vdtheta,vsdtheta = 0,0#deg_rad(1), deg_rad(.5)

vD, svD = wa_Ds/1e3, swa_Ds/1e3
vd, svd = wa_drail/1e3, swa_drail/1e3

g = fg(va,vtheta,vdtheta,vD,vd)
print(fg(va,vtheta,vdtheta,vD,vd))
sg = fsg(va,vsa,vtheta,vstheta,vdtheta,vsdtheta,vD,vsD,vd,vsd)