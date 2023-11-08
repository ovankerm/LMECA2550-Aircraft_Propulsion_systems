# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:08:56 2019

@author: Romain Debroeyer
"""
import numpy as np
import scipy.interpolate as sp
# Look-up for the behavior of the NACA16-509

def naca16_509_m06(a, filename):
    with open(filename,"r") as f:
        f.readline()
        f1 = f.readlines()
        alphas_naca16_509_m06 = np.zeros(np.size(f1))
        cls_naca16_509_m06 = np.zeros(np.size(f1))
        cds_naca16_509_m06 = np.zeros(np.size(f1))
        for i in np.arange(np.size(f1)):
            dat = f1[i].split()
            alphas_naca16_509_m06[i] = dat[0]
            cds_naca16_509_m06[i] = dat[1]
            cls_naca16_509_m06[i] = dat[2]
        
        a = 180/np.pi*a
        fcl = sp.interp1d(alphas_naca16_509_m06,cls_naca16_509_m06,'cubic',bounds_error=False)
        cl = fcl(a)
        # Polynomial fits
        
        
        if cl!=cl:
            #p1 = polyfit([alphas_naca16_509_m06(1) -80 -90 -100], [-0.246163682864450 -0.02 0 -0.02] , 3);
            p1 = (0.000001993645465, 0.000338284275449, 0.012246220243973, -0.184575265468565)
            #p2 = polyfit([alphas_naca16_509_m06[-1] 12 70 90 110], [0.966112531969310 1.1 0.1 0 0.1] , 4);
            p2 = (-0.000000079103690, 0.000023256824678, -0.002153261849550, 0.055200854989657, 0.709111947532640)
            
            if a<0:
                cl = (-0.246163682864450) 
                cl = (np.polyval(p1,a))
            else:
                cl = (0.966112531969310)
                cl = (np.polyval(p2,a))
            
        fcd = sp.interp1d(alphas_naca16_509_m06,cds_naca16_509_m06,'cubic',bounds_error=False)
        cd = fcd(a)
        
        if cd != cd:
            coef1 =  -0.067317398298573
            coef2 =  0.023120640006175
            if a<0:
                cd = cds_naca16_509_m06[0]-0.5 * coef1 * (a-alphas_naca16_509_m06[0])**2
            else:
                cd = cds_naca16_509_m06[-1] + 0.5 * coef2 * (a-alphas_naca16_509_m06[-1])**2
            
        cdmax = 2
        cd = min(cd,cdmax)
        return cl,cd