#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 18:12:35 2018

@author: nephilim
"""
import numpy as np
from matplotlib import pyplot,cm
z_site_list=[18,28,38,48,58,68,78,88,98,108,118,128,138,148,158,168,178,188,198,208,218]
for z_site in z_site_list:
    locals()['pdata_%s'%z_site]=np.load('/home/nephilim/Python_code/FDTD_2D/FDTD_2D_numba_Pool_Class/%s_pdata.npy'%z_site)
for z_site in z_site_list:
    locals()['record_%s'%z_site]=np.load('/home/nephilim/Python_code/FDTD_2D/FDTD_2D_numba_Pool_Class/%s_record.npy'%z_site)

data=np.load('./forward_data_file/58_P_data.npy')
for index,p in enumerate(data):
    if index==0:
        fig=pyplot.figure(1)
        image=pyplot.imshow(p,animated=True,cmap=cm.seismic,interpolation='nearest',vmin=-0.05,vmax=0.05)
        pyplot.colorbar()
    else:
        image.set_data(p)
        pyplot.pause(0.001)
        fig.canvas.draw()


image=pyplot.imshow(record_158,animated=True,cmap=cm.seismic,interpolation='nearest',vmin=-0.05,vmax=0.05)
