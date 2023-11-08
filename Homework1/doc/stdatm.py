#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 08:16:08 2020

@author: Romain Debroeyer
MECA2550
Standard atmosphere
"""


def stdatm(z):
    """
    Parameters
    ----------
    z : altitude [m]

    Returns
    -------
    p [Pa], rho [kg/m^3], T [K], gamma, R [J/kg/K], cp [J/kg/K]

    """    
    T0 = 288.15 # K
    g = 9.81 #	m/s2
    l = -6.5E-3 #	K/m
    gamma =  1.4 #
    R = 287.0529 # J/kg/K
    cp = 1005 # J/kg/K
    gammad = -9.8E-3 # K/m
    rho0 = 1.225 # kg/m3
    p0 = 101325 # Pa
    
    
    T = (1 + l*z/T0) * T0
    
    p = ((T/T0)**(-g/(R*l))) * p0
    
    rho = ((T/T0)**(-g/(R*l) -1 )) * rho0
    
    p0 / (rho0*T0)
    
    p - rho*R*T
    
    p0 - rho0*R*T0

    return p,rho,T,gamma,R,cp