# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 20:19:14 2020

@author: klein
"""

import sympy
from sympy import *
import math
g,a,theta,dtheta,D,d = sympy.symbols("g,a,theta,dtheta,D,d")
sa,stheta,sdtheta,sD,sd,sg = sympy.symbols("sigma_a, sigma_theta, sigma_dtheta, sigma_D, sigma_d, sigma_g")

g = a/sin(theta+dtheta)*(1+2/5*D**2/(D**2-d**2))
sg = sqrt((g.diff(a) * sa)**2 + (g.diff(theta) * stheta)**2 + 
          (g.diff(dtheta) * sdtheta)**2+ (g.diff(D) * sD)**2+
          (g.diff(d) * sd)**2)
fg = sympy.lambdify((a,theta,dtheta,D,d),g, ("math", "mpmath", "sympy"))
fsg = sympy.lambdify((a,sa,theta,stheta,dtheta,sdtheta,D,sD,d,sd),
                     sg,("math", "mpmath", "sympy"))
va, vsa = 2,.1
vtheta, vstheta = 10,1#deg_rad(30), deg_rad(1) 
vdtheta,vsdtheta = 1,0.1
#deg_rad(1), deg_rad(.5)
vD, vsD = .03, .001
vd, vsd = .01, .001
# Numerically evaluate expressions and print 
vg = fg(va,vtheta,vdtheta,vD,vd)

display(g)
display(latex(g))