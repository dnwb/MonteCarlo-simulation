#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:47:49 2019

Monte carlo realization simulation

Take Geomagnetic field M and production rate Q to calculate
the solar modulation potential using monte carlo 

@author: wu
"""

import numpy


def closest(number, array, lower=None, upper=None):
    if lower == None and upper == None:
        idx = (numpy.abs(number-array)).argmin() 
    elif lower is not None:
        idx = numpy.where(array == array[array <= number].max())[0]
    elif upper is not None:
        idx = numpy.where(array == array[array >= number].min())[0]
    return idx


# Monte carlo main body
#-----------------------------------------------------------------------------
def montecarlo(year, realizationM, Q_fin, phi_fin, M_fin, Q, dQ, resolution):
    
    N = len(Q)
    
    # initial setting
    #-------------------------------------------------
    if resolution == 'high':
        phi_fin2 = numpy.linspace(0, 2.0, 1001)
    elif resolution == 'low':
        phi_fin2 = numpy.linspace(0, 2.0, 101)
    nphi = len(phi_fin2)

    Q_fin2 = numpy.zeros(shape=(151, nphi))
    for i in range(151):
        Q_fin2[i,:] = numpy.interp(phi_fin2, phi_fin, Q_fin[i,:])
    
    # Output results
    #-------------------------------------------------
    phi_bestfit = [0]*N
    phi_low = [0]*N
    phi_upp = [0]*N
   # chi2total = numpy.zeros(shape=(nphi,N))
   # bestchi2 = [0]*N
    
    # running realizations
    #-------------------------------------------------
    for k in range(N): # each time step N
        
        if resolution == 'high':    
            M_realization = realizationM[k,:]
            Q_realization = numpy.random.normal(size=1000)*dQ[k]+Q[k]
        elif resolution == 'low':
            a = numpy.int_(numpy.random.uniform(0,1,100)*999)
            M_realization = realizationM[k,a]
            Q_realization = numpy.random.normal(size=100)*dQ[k]+Q[k]
            
        monte = 0
        if resolution == 'high':
            phi_star = [0]*1000000
        elif resolution == 'low':
            phi_star = [0]*10000
        
        if resolution == 'high':
            jmn = 1000
            iqn = 1000
        elif resolution == 'low':
            jmn = 100
            iqn = 100
        
        for jm in range(jmn):
            Mt = M_realization[jm]
    
            low_idx = closest(Mt, M_fin, lower='lower')
            upp_idx = closest(Mt, M_fin, lower='upper')
            Mlow = M_fin[low_idx]
            Mupp = M_fin[upp_idx]
            Msub = numpy.linspace(Mlow, Mupp, 101)
            Mclose = closest(Mt, Msub)
            
            Qarray = [0]*nphi
    
            for l in range(nphi):
                
                Qlow=Q_fin2[low_idx, l]
                Qupp=Q_fin2[upp_idx, l]
                
                Qsub = numpy.linspace(Qlow, Qupp, 101)
                Qarray[l] = Qsub[Mclose]
    
            for iq in range(iqn):
                Qt = Q_realization[iq]
                Qclose = closest(Qt, Qarray)
                phi_star[monte] = phi_fin2[Qclose]
                monte=monte+1
        
        if resolution == 'high':
            Q_star = [0]*1000000
        elif resolution == 'low':
            Q_star = [0]*10000
            
        if resolution == 'high':
            pn = 1000000
        elif resolution == 'low':
            pn = 10000
            
        for p in range(pn):
            ph = phi_star[p]
            
            Phiclose = closest(ph, phi_fin2, lower='lower')
            Qarray = Q_fin2[:,Phiclose]
            Q_star[p] = Qarray[Mclose]
    
        chi2 = [0]*nphi
        Qarray = Q_fin2[Mclose,:]
        
        for h in range(nphi):
            Q2 = Qarray[h]
            chi2[h]=( (numpy.mean(Q_star)-Q2)/numpy.std(Q_star, ddof=1) )**2
        
        g = numpy.where(chi2 == min(chi2))
        g = int(g[0])
        phi_bestfit[k] = phi_fin2[g]
    
        chi2err = min(chi2)+1
        vor = chi2[0:g+1]
        nach = chi2[g:nphi]
    
        idx = (numpy.abs(chi2err-vor)).argmin()
        phi_low[k] = phi_fin2[idx]
        idx = (numpy.abs(chi2err-nach)).argmin()
        phi_upp[k] = phi_fin2[idx+g]

    return phi_bestfit, phi_low, phi_upp

