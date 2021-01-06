#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:58:37 2018

@author: nephilim
"""
from numba import jit
import numpy as np

#
@jit(nopython=True)
def Abnormal_Model(xl,zl,CPML):
    epsilon=np.ones((xl+2*CPML,zl+2*CPML))*4
    sigma=np.ones((xl+2*CPML,zl+2*CPML))*0.003
    p=34+CPML
    l=14
    w=3
    for i in range(50,70):
        for j in range(50,70):
            epsilon[i][j]=1
            sigma[i][j]=0.1
    return epsilon,sigma

@jit(nopython=True)
def Create_iModel(xl,zl,CPML):
    epsilon=np.ones((xl+2*CPML,zl+2*CPML))*4
    sigma=np.ones((xl+2*CPML,zl+2*CPML))*0.003
    return epsilon,sigma