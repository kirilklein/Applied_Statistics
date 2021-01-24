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
from scipy import optimize
# In[Import custom]:
basepath = os.path.abspath('')
dpath = f"{basepath}/Incline"
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure

# In[Define functions]:
def z_value(arr): return (arr-np.mean(arr))/np.std(arr)
def std_to_prob(std): return 1-(stats.norm.cdf(abs(std))-stats.norm.cdf(-abs(std)))
def chauvenet_prob(arr): 
    z = z_value(arr)
    prob = len(arr)*std_to_prob(z)
    return prob
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

def d_theta(a1,a2,a,b): return (a1-a2)*b/((a1+a2)*a)
def sig_theta_cc(a,sa,b,sb): return 1/(a**2+b**2)*np.sqrt((b*sa)**2+sb**2)
def sig_dtheta(a1,sa1,a2,sa2,a,sa,b,sb):
    den = 1/(a1+a2)
    fda1 = b/a*2*a2*den**2
    fda2 = b/a*2*a1*den**2
    fda = b/a**2*(a1-a2)*den
    fdb = 1/a*(a1-a2)*den
    sig_dtheta = np.sqrt((sa1*fda1)**2+(sa2*fda2)**2+(sa*fda)**2+(sb*fdb)**2)
    return sig_dtheta
def chi_sq(arr,sig_arr): return np.sum((arr-cf.weighted_avg(arr,sig_arr))**2/sig_arr**2)
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
fsg = sympy.lambdify((a,sa,theta,stheta,dtheta,sdtheta,D,sD,d,sd),
                     sg,("math", "mpmath", "sympy"))

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
tl_1b = (tl_1b.T- tl_1b[:,0]).T
tl_1s = (tl_1s.T-tl_1s[:,0]).T
tl_2b = (tl_2b.T-tl_2b[:,0]).T
tl_2s = (tl_2s.T-tl_2s[:,0]).T
print(tl_1s)
# In[just visualization]:

xlab, ylab = 'Time (s)', 'Channel 1 (V)' 
files = fn_m1b
data = Time1b
times = tl_1b

fig, ax = plt.subplots(len(files))
for i in range(len(files)):
    ax[i].plot(data[i][xlab]-tl_1b[i,0], data[i][ylab])
    ax[i].vlines(times[i],ymin = 0, ymax = 3,colors = 'r')
    #ax[i].set_xlim((-.8,.7))
plt.show()

# In[Store data in array]
data_all = pd.read_csv("project_data.csv", skiprows = 2)
indices_angle = np.arange(15)
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

Delta_theta_gonio = (wa_angle-wa_angle_r)/2
sDelta_theta_gonio = 1/2*np.sqrt(swa_angle**2+swa_angle_r**2)
print(Delta_theta_gonio, sDelta_theta_gonio)

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
a_arr = data_all['a'][1:5]
b_arr = data_all['b'][1:5]
#c_arr = data_all['c'][1:4]
mu_a = np.mean(data_all['a'][1:5])
mu_b = np.mean(data_all['b'][1:5])
mu_c = np.mean(data_all['c'][1:4])
mu_c_comp = np.sqrt(a_arr**2+b_arr**2)
diff_c = mu_c_comp-mu_c
theta_trig = 360*np.arctan(b_arr/a_arr)/(np.pi*2)


s_abc = .05*np.ones(4)
wa_mua = weighted_avg(mu_a, s_abc)
wa_mub = weighted_avg(mu_b, s_abc)
chi2_al = chi_sq(a_arr,s_abc)
chi2_b = chi_sq(b_arr,s_abc)
prob_a = stats.chi2.cdf(chi2_al, 3)
prob_b = stats.chi2.cdf(chi2_b, 3)
# In[Determine acceleration]

a_1b, sa_1b, args_1b, popt_1b = [],[], [],[]
a_1s, sa_1s, args_1s = [],[], []
a_2b, sa_2b, args_2b = [],[], []
a_2s, sa_2s, args_2s = [],[], []
Prob_1b, Prob_1s, Prob_2b, Prob_2s = [],[],[],[]
Chi2_1b, Chi2_1s, Chi2_2b, Chi2_2s = [],[],[],[]
for i in range(len(tl_1b)):
    minuit_obj,Chi2, Prob = minuit_fit(quadr, tl_1b[i],wa_P, swa_P )
    a = minuit_obj.values['a']
    sa = minuit_obj.errors['a']
    a_1b.append(a)
    sa_1b.append(sa)
    args_1b.append(minuit_obj.args)
    Chi2_1b.append(Chi2)
    Prob_1b.append(Prob)
for i in range(len(tl_1s)):
    minuit_obj,Chi2, Prob = minuit_fit(quadr, tl_1s[i],wa_P, swa_P )
    a = minuit_obj.values['a']
    sa = minuit_obj.errors['a']
    a_1s.append(a)
    sa_1s.append(sa)
    args_1s.append(minuit_obj.args)
    Chi2_1s.append(Chi2)
    Prob_1s.append(Prob)
for i in range(len(tl_2b)):
    minuit_obj,Chi2, Prob = minuit_fit(quadr, tl_2b[i],wa_P, swa_P )
    a = minuit_obj.values['a']
    sa = minuit_obj.errors['a']
    a_2b.append(a)
    sa_2b.append(sa)
    args_2b.append(minuit_obj.args)
    Chi2_2b.append(Chi2)
    Prob_2b.append(Prob)
for i in range(len(tl_2s)):
    minuit_obj,Chi2, Prob = minuit_fit(quadr, tl_2s[i],wa_P, swa_P )
    a = minuit_obj.values['a']
    sa = minuit_obj.errors['a']
    a_2s.append(a)
    sa_2s.append(sa)
    args_2s.append(minuit_obj.args)   
    Chi2_2s.append(Chi2)
    Prob_2s.append(Prob)
a_1b,a_1s,a_2b,a_2s = np.array(a_1b), np.array(a_1s), np.array(a_2b), np.array(a_2s)
sa_1b,sa_1s,sa_2b,sa_2s = np.array(sa_1b), np.array(sa_1s), np.array(sa_2b), np.array(sa_2s)

z_a1b, z_a1s = z_value(a_1b), z_value(a_1s)
z_a2b, z_a2s = z_value(a_2b), z_value(a_2s)
p_a1b, p_a1s = len(z_a1b)*std_to_prob(z_a1b), len(z_a1s)*std_to_prob(z_a1s)
p_a2b, p_a2s = len(z_a2b)*std_to_prob(z_a2b), len(z_a2s)*std_to_prob(z_a2s)


wa_a_1b = weighted_avg(a_1b,sa_1b)
wa_a_1s = weighted_avg(a_1s,sa_1s)
wa_a_2b = weighted_avg(a_2b,sa_2b)
wa_a_2s = weighted_avg(a_2s,sa_2s)

fig,ax = plt.subplots()
ax.plot(tl_2s.T,wa_P,'.')
x = np.linspace(-.8,0.6,100)
for i in range(len(tl_2s)):
    ax.plot(x, quadr(x,*args_2s[i]))
print(a/100,sa/100)
# In[Crosse check angle]
wa_a_1b = weighted_avg(a_1b,sa_1b)
swa_a_1b = sig_weighted_avg(sa_1b)

wa_a_1s = weighted_avg(a_1s,sa_1s)
swa_a_1s = sig_weighted_avg(sa_1s)
wa_a_2s = weighted_avg(a_2s,sa_2s)
swa_a_2s = sig_weighted_avg(sa_2s)
wa_a_2b = weighted_avg(a_2b,sa_2b)
swa_a_2b = sig_weighted_avg(sa_2b)

dr_const = 360/(2*np.pi)

delta_theta_b = dr_const*d_theta(wa_a_1b, wa_a_2b,mu_a, mu_b)
s_delta_theta_b = dr_const*sig_dtheta(wa_a_1b,swa_a_1b,
                           wa_a_2b,swa_a_2b,wa_mua,s_abc,wa_mub,s_abc)

delta_theta_s = dr_const*d_theta(wa_a_1s, wa_a_2s, wa_mua, wa_mub)
s_delta_theta_s = dr_const*sig_dtheta(wa_a_1s,swa_a_1s,
                           wa_a_2s,swa_a_2s,wa_mua,s_abc,wa_mub,s_abc)
wa_delta_theta_s = weighted_avg(delta_theta_s,s_delta_theta_s)
swa_delta_theta_s = sig_weighted_avg(s_delta_theta_s)

wa_delta_theta_b = weighted_avg(delta_theta_b,s_delta_theta_b)
swa_delta_theta_b = sig_weighted_avg(s_delta_theta_b)

delta_theta_all = np.array([wa_delta_theta_s,wa_delta_theta_b])
sdelta_theta_all = np.array([swa_delta_theta_s,swa_delta_theta_b])
wa_dtheta_all = weighted_avg(delta_theta_all,sdelta_theta_all)
swa_dtheta_all = sig_weighted_avg(sdelta_theta_all)

theta_cc = dr_const*np.arctan(mu_b/mu_a)
s_theta_cc = dr_const*sig_theta_cc(mu_a,s_abc[0],mu_b,s_abc[0])
print(theta_cc-wa_dtheta_all)
#print(theta_cc, s_theta_cc, delta_theta_s, s_delta_theta_s)
# In[Compute g]


va, vsa = wa_a_1b/100, np.sqrt(1/np.sum(1/sa_1b**2))/100
print(va)
vtheta, vstheta = deg_rad(wa_angle), 2*np.pi*deg_rad(swa_angle)/360 
vdtheta,vsdtheta = 0,0#deg_rad(1), deg_rad(.5)
vD, svD = wa_Db/1e3, swa_Db/1e3
vd, svd = wa_drail/1e3, swa_drail/1e3

#gv =  g_val(wa_a_1b/100,deg_rad(wa_angle),0,wa_Db/1e3,wa_drail/1e3)

g1b = fg(va,vtheta,vdtheta,vD,vd)
print(fg(va,vtheta,vdtheta,vD,vd))
sg1b = fsg(va,vsa,vtheta,vstheta,vdtheta,vsdtheta,vD,svD,vd,svd)
# In[]



va, vsa = wa_a_1s/100, np.sqrt(1/np.sum(1/sa_1s**2))/100
print(va)
vtheta, vstheta = deg_rad(wa_angle), 2*np.pi*deg_rad(swa_angle)/360 
vdtheta,vsdtheta = 0,0#deg_rad(1), deg_rad(.5)

vD, svD = wa_Ds/1e3, swa_Ds/1e3
vd, svd = wa_drail/1e3, swa_drail/1e3

#gv =  g_val(wa_a_1s/100,deg_rad(wa_angle),0,wa_Ds/1e3,wa_drail/1e3)
g1s = fg(va,vtheta,vdtheta,vD,vd)
print(fg(va,vtheta,vdtheta,vD,vd))
sg1s = fsg(va,vsa,vtheta,vstheta,vdtheta,vsdtheta,vD,svD,vd,svd)

# In[]



va, vsa = wa_a_2s/100, np.sqrt(1/np.sum(1/sa_2s**2))/100
print(va)
vtheta, vstheta = deg_rad(wa_angle_r), 2*np.pi*deg_rad(swa_angle_r)/360 
vdtheta,vsdtheta = 0,0#deg_rad(1), deg_rad(.5)

vD, svD = wa_Ds/1e3, swa_Ds/1e3
vd, svd = wa_drail/1e3, swa_drail/1e3

g2s = fg(va,vtheta,vdtheta,vD,vd)
print(fg(va,vtheta,vdtheta,vD,vd))
sg2s = fsg(va,vsa,vtheta,vstheta,vdtheta,vsdtheta,vD,svD,svd,svd)
# In[]


va, vsa = wa_a_2b/100, np.sqrt(1/np.sum(1/sa_2b**2))/100
print(va)
vtheta, vstheta = deg_rad(13.5),0#deg_rad(wa_angle_r), 2*np.pi*deg_rad(swa_angle_r)/360 
vdtheta,vsdtheta = 0,0#deg_rad(1), deg_rad(.5)

vD, svD = wa_Db/1e3, swa_Db/1e3
vd, svd = wa_drail/1e3, swa_drail/1e3

g2b = fg(va,vtheta,vdtheta,vD,vd)
print(fg(va,vtheta,vdtheta,vD,vd))
sg2b = fsg(va,vsa,vtheta,vstheta,vdtheta,vsdtheta,vD,svD,vd,svd)

# In[avg over g]
g_arr = np.array([float(g1s),float(g1b),float(g2s),float(g2b)])
sg_arr = np.array([float(sg1s),float(sg1b),float(sg2s),float(sg2b)])
g_final = weighted_avg(g_arr, sg_arr)
sg_final = sig_weighted_avg(sg_arr)
print(g_final, sg_final)

# In[Plot]

xlab, ylab = 'Time (s)', 'Channel 1 (V)' 
files = fn_m1b
data = Time1b
times = tl_1b

fig, ax = plt.subplots()
ax.plot(data[0][xlab]+tl_1b[0,:], data[0][ylab])
ax.vlines(times[0],ymin = 0, ymax = 3,colors = 'r')
    #ax[i].set_xlim((-.8,.7))
plt.show()