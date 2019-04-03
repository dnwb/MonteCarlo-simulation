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
    '''find the index in an array that is closest to a given number'''
    if lower is None and upper is None:
        idx = (numpy.abs(number-array)).argmin()
    elif lower is not None:
        idx = numpy.where(array == array[array <= number].max())[0]
    elif upper is not None:
        idx = numpy.where(array == array[array >= number].min())[0]
    return idx


def qarray_func(n_phi, q_fin, lowind, uppind, m_close):
    '''calculate Qarray'''
    result = [numpy.linspace(q_fin[lowind, x], q_fin[uppind, x], 101)[m_close] for x in range(n_phi)]
    return result


def inner_mc_func(iqn, q_realization, q_arr, p_fin):
    ''' inner monte carlo function'''
    result = [p_fin[closest(q_realization[x], q_arr)] for x in range(iqn)]
    return numpy.reshape(result, (iqn, 1))


# it is not faster
def make_q_star(pn, q_fin, p_star, p_fin, m_close):
    '''calculating Q_star'''
    result = [q_fin[:, closest(p_star[x], p_fin, lower='lower')][m_close] for x in range(pn)]
    return result


def chi_func(array, star):
    '''calculating the chi2'''
    result = [((numpy.mean(star)-x)/numpy.std(star, ddof=1))**2 for x in array]
    return result


# Monte carlo main body
#-----------------------------------------------------------------------------
def montecarlo(realization_M, Q_fin, Q, dQ, resolution, realization_Q=None):
    '''Monte carlo realization simulation'''

    phi_fin = numpy.linspace(0, 2.0, 101)
    M_fin = numpy.linspace(0, 15.0, 151)
    N = len(Q)

    # initial setting
    if resolution == 'high':
        phi_fin2 = numpy.linspace(0, 2.0, 1001)
    elif resolution == 'low':
        phi_fin2 = numpy.linspace(0, 2.0, 101)
    nphi = len(phi_fin2)

    Q_fin2 = numpy.asarray([numpy.interp(phi_fin2, phi_fin, Q_fin[x, :]) for x in range(151)])

    # Output results
    #-------------------------------------------------
    phi_bestfit = [0]*N
    phi_low = [0]*N
    phi_upp = [0]*N
    chi2total = numpy.zeros(shape=(nphi, N))

    # each time step N
    #-------------------------------------------------
    for k in range(N):
        print(f'{k}/{N}')

        # 14C provides its own realizations
        if realization_Q is None:
            if resolution == 'high':
                M_realization = realization_M[k, :]
                Q_realization = numpy.random.normal(size=1000)*dQ[k]+Q[k]

                phi_star = numpy.zeros(shape=(1000000, 1))
                Q_star = numpy.zeros(shape=(1000000, 1))
                jmn = 1000
                iqn = 1000
                pn = 1000000

            elif resolution == 'low':
                tmp = numpy.int_(numpy.random.uniform(0, 1, 100)*999)
                M_realization = realization_M[k, tmp]
                Q_realization = numpy.random.normal(size=100)*dQ[k]+Q[k]

                phi_star = numpy.zeros(shape=(10000, 1))
                Q_star = numpy.zeros(shape=(10000, 1))
                jmn = 100
                iqn = 100
                pn = 10000

        elif realization_Q is not None:
            if resolution == 'high':
                M_realization = realization_M[k, :]
                Q_realization = realization_Q[k, :]

                phi_star = numpy.zeros(shape=(1000000, 1))
                Q_star = numpy.zeros(shape=(1000000, 1))
                jmn = 1000
                iqn = 1000
                pn = 1000000

            elif resolution == 'low':
                tmp = numpy.int_(numpy.random.uniform(0, 1, 100)*999)
                M_realization = realization_M[k, tmp]
                tmp = numpy.int_(numpy.random.uniform(0, 1, 100)*999)
                Q_realization = realization_Q[k, tmp]

                phi_star = numpy.zeros(shape=(10000, 1))
                Q_star = numpy.zeros(shape=(10000, 1))
                jmn = 100
                iqn = 100
                pn = 10000

        monte = 0
        for jm in range(jmn):
            Mt = M_realization[jm]

            low_idx = closest(Mt, M_fin, lower='lower')
            upp_idx = closest(Mt, M_fin, upper='upper')
            Mlow = M_fin[low_idx]
            Mupp = M_fin[upp_idx]
            Msub = numpy.linspace(Mlow, Mupp, 101)
            Mclose = closest(Mt, Msub)

            Qarray = qarray_func(nphi, Q_fin2, low_idx, upp_idx, Mclose)
            
            phi_star[monte: monte+iqn] = inner_mc_func(iqn, Q_realization, Qarray, phi_fin2)
            monte = monte+iqn

        for p in range(pn):
            phi_close = closest(phi_star[p], phi_fin2, lower='lower')
            Qarray = Q_fin2[:, phi_close]
            Q_star[p] = Qarray[Mclose]

        chi2 = [0]*nphi
        Qarray = Q_fin2[Mclose, :]

        chi2 = chi_func(Qarray, Q_star)

        tmp = numpy.where(chi2 == min(chi2))
        ind = int(tmp[0])
        phi_bestfit[k] = phi_fin2[ind]

        chi2err = min(chi2)+1
        vor = chi2[0:ind+1]
        nach = chi2[ind:nphi]

        idx = (numpy.abs(chi2err-vor)).argmin()
        phi_low[k] = phi_fin2[idx]
        idx = (numpy.abs(chi2err-nach)).argmin()
        phi_upp[k] = phi_fin2[idx+ind]

    chi2total[:, k] = chi2

    return phi_bestfit, phi_low, phi_upp, chi2total
