#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:33:41 2018

@author: nephilim
"""
import Create_Model
import Forward2D
import Calculate_Gradient_NoSave_Pool
import numpy as np
import time
from pathlib import Path
#import scipy.io as sio
from Optimization import para,options,Optimization
from matplotlib import pyplot,cm
from skimage import filters



if __name__=='__main__':       
    para.xl=101
    para.zl=101
    para.dx=0.1
    para.dz=0.1
    para.k_max=1500
    para.dt=1e-10
    para.ricker_freq=2e8
    para.beta=1
    para.eta0=377
    para.lambda_=0.5
    para.CPML=10
    para.Npower=4
    para.k_max_CPML=5
    para.alpha_max_CPML=0.008
    para.Rcoef=1e-8
    epsilon,sigma=Create_Model.Abnormal_Model(para.xl,para.zl,para.CPML)
    pyplot.figure(1)
    pyplot.subplot(121)
    pyplot.imshow(epsilon,cmap=cm.jet,vmax=10,vmin=1)
    pyplot.colorbar()
    pyplot.subplot(122)
#    pyplot.figure(2)
    pyplot.imshow(sigma,cmap=cm.jet,vmax=0.01,vmin=0)
    pyplot.colorbar()
#    para.rho=Create_Model(para.xl,para.zl)
#    vp=np.zeros((para.xl+9*2,para.zl+9*2))
#    vp_temp=sio.loadmat('Vp_data.mat')['vp']*10
#    vp[29:29+para.xl-40,29:29+para.zl-40]=vp_temp
#    vp[0:29]=vp[29]
#    vp[29+para.xl-40:]=vp[28+para.xl-40]
#    vp[:,0:29]=vp[:,29][:,np.newaxis]
#    vp[:,29+para.zl-40:]=vp[:,28+para.zl-40][:,np.newaxis]

    x_site=[para.CPML]*101
    z_site=np.arange(10,111)
    para.source_site=np.column_stack((x_site,z_site))    
    para.data=[]
    
    ref_pos_x=[para.CPML]*101
    ref_pos_z=np.arange(10,111)
    para.ref_pos=np.column_stack((ref_pos_x,ref_pos_z))    
#    pyplot.imshow(vp,cmap=cm.seismic)
    # pyplot.plot(ref_pos_x,ref_pos_z,'x')
    # pyplot.plot(x_site,z_site,'rp')
#    pyplot.colorbar()
#    pyplot.figure()
#    fh=lambda x:Calculate_Gradient_NoSave.misfit(x,para)

    freq = [0.5e8,1e8]
    maxiter = [20,20]
    tol = [1e-2,1e-3]
    lambda_=[0,0]
#    for ii in range(len(freq)):
    start_time=time.time()
    for k in range(len(freq)):
        # if k==1:
        #     continue
        para.ricker_freq=freq[k]
        para.lambda_=lambda_[k]
        start_time1=time.time()
#       sigma = sigma
        Forward2D.Forward_2D(epsilon,sigma*para.eta0/para.beta,para)
        print('Forward Done !')
        print('Elapsed time is %s seconds !'%str(time.time()-start_time1))
        
        data=np.load('./forward_data_file/Forward_data.npy')
        para.data=data
        
        
        
        fh=lambda x,y:Calculate_Gradient_NoSave_Pool.misfit(x,y,para)
        
        
#            pyplot.figure(2)
#            pyplot.imshow(para.data[9],cmap=cm.seismic,vmin=-10, vmax=10)
#            pyplot.colorbar()
        if k==0:
            iepsilon,isigma=Create_Model.Create_iModel(para.xl,para.zl,para.CPML)
            isigma=isigma*para.eta0/para.beta
        else:
            dir_path='./imodel_file'
            file_num=int(len(list(Path(dir_path).iterdir()))/2)
            data=np.load('./imodel_file/%s_imodel.npy'%file_num)
            iepsilon=data[:int(len(data)/2)].reshape((121,-1))
            isigma=data[int(len(data)/2):].reshape((121,-1))
        
        # f,g=fh(iepsilon,isigma)
        # pyplot.figure()
        # pyplot.imshow(g[:int(len(g)/2)].reshape((121,121)))
        # pyplot.figure()
        # pyplot.imshow(g[int(len(g)/2):].reshape((121,121)))

        options.method='lbfgs'
        options.tol=tol[k]
        options.maxiter=maxiter[k]
        Optimization_=Optimization(fh,iepsilon,isigma)
        imodel,info=Optimization_.optimization()
        print('Elapsed time is %s seconds !'%str(time.time()-start_time1))
    
        epsilon_inv = imodel[:int(len(imodel)/2)].reshape((para.xl+2*para.CPML,-1))
        sigma_inv = imodel[int(len(imodel)/2):].reshape((para.xl+2*para.CPML,-1))*para.beta/para.eta0
        pyplot.figure()
        pyplot.subplot(121)
        pyplot.imshow(epsilon_inv,cmap=cm.jet,vmax=10,vmin=1)
        pyplot.colorbar()
        pyplot.subplot(122)
#    pyplot.figure(5)
        pyplot.imshow(sigma_inv,cmap=cm.jet,vmax=0.01,vmin=0)
        pyplot.colorbar()
        data_=[]
        for info_ in info:
            data_.append(info_[3])
        pyplot.figure(6)
        pyplot.plot(data_/data_[0])
        pyplot.yscale('log')
    print('Elapsed time is %s seconds !'%str(time.time()-start_time))