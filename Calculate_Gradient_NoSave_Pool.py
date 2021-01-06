#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:43 2018

@author: nephilim
"""
from multiprocessing import Pool
import numpy as np
import time
import Add_CPML
import Wavelet
import Time_loop
import Reverse_time_loop

def calculate_gradient(epsilon,sigma,index,CPML_Params,para):
    k_max=para.k_max
    ep0=8.841941282883074e-12
    t=np.arange(k_max)*para.dt
    f=Wavelet.ricker(t,para.ricker_freq)
    
    data=para.data[:,index]

    z_site_index,V_data,idata=Time_loop.time_loop(para.xl,para.zl,para.dx,para.dz,para.dt,\
                                                  epsilon,sigma,CPML_Params,f,k_max,\
                                                  para.source_site[index],para.ref_pos[index])
    rhs_data=idata-data
    
    RT_P_data=Reverse_time_loop.reverse_time_loop(para.xl,para.zl,para.dx,para.dz,\
                                                  para.dt,epsilon,sigma,CPML_Params,k_max,\
                                                  para.ref_pos[index],rhs_data)

    time_sum_eps=np.zeros((para.xl+2*CPML_Params.npml,para.zl+2*CPML_Params.npml))
    time_sum_sig=np.zeros((para.xl+2*CPML_Params.npml,para.zl+2*CPML_Params.npml))
#    dudx=np.zeros((para.xl+2*CPML_Params.CPML,para.zl+2*CPML_Params.CPML))
#    dudz=np.zeros((para.xl+2*CPML_Params.CPML,para.zl+2*CPML_Params.CPML))
#    dpdx=np.zeros((para.xl+2*CPML_Params.CPML,para.zl+2*CPML_Params.CPML))
#    dpdz=np.zeros((para.xl+2*CPML_Params.CPML,para.zl+2*CPML_Params.CPML))
    for k in range(1,k_max-1):
        u1=V_data[k+1]
        u0=V_data[k-1]
        u=V_data[k]
        p1=RT_P_data[k]
        time_sum_eps+=p1*(u1-u0)/para.dt/2
        time_sum_sig+=p1*u
        
#        u=V_data[k]
#        dudx[1:u.shape[0]-1,:]=(u[2:u.shape[0],:]-u[:u.shape[0]-2,:])/para.dx/2
#        dudz[:,1:u.shape[1]-1]=(u[:,2:u.shape[1]]-u[:,:u.shape[1]-2])/para.dz/2
#        dpdx[1:p1.shape[0]-1,:]=(p1[2:p1.shape[0],:]-p1[:p1.shape[0]-2,:])/para.dx/2
#        dpdz[:,1:p1.shape[1]-1]=(p1[:,2:p1.shape[1]]-p1[:,:p1.shape[1]-2])/para.dz/2

    g_eps=ep0*time_sum_eps
    g_sig=para.beta*time_sum_sig/para.eta0
#    g[0:9+CPML+5][:]=0
#    return rhs_data.flatten(),g.flatten()
    return rhs_data.flatten(),g_eps.flatten(),g_sig.flatten()    

def calculate_toltal_variation_model(rho,vp,dx,dz):
    normagra_rho,g_rho_TolVar=cal_toltal_variation(rho,dx,dz)
    normagra_vp,g_vp_TolVar=cal_toltal_variation(vp,dx,dz)
    f_TolVar=np.sum(normagra_rho.flatten())+np.sum(normagra_vp.flatten())
    g_TolVar=np.hstack((g_rho_TolVar.flatten(),g_vp_TolVar.flatten()))
    return f_TolVar,g_TolVar

def cal_toltal_variation(u,dx,dz):
    epsilon=1e-8
    dudx=(np.vstack((u[1:,:],u[-1,:]))-u)/dx
    dudz=(np.hstack((u[:,1:],u[:,-1][:,np.newaxis]))-u)/dz
    normgrad=np.sqrt(dudx**2+dudz**2+epsilon)
    fx=dudx/normgrad
    fz=dudz/normgrad
    gx=(fx-np.vstack((fx[0,:],fx[:-1,:])))/dx
    gz=(fz-np.hstack((fz[:,0][:,np.newaxis],fz[:,:-1])))/dz
    div=-(gx+gz)
    return normgrad,div

def misfit(epsilon,sigma,para): 
    lambda_=para.lambda_
    start_time=time.time()  
    sigma = para.beta*sigma/para.eta0
#    vp_max=max(vp.flatten())
    CPML_Params=Add_CPML.Add_CPML(para.xl,para.zl,para.CPML,epsilon,sigma,para.dx,para.dz,para.dt,\
                                  para.Npower,para.k_max_CPML,para.alpha_max_CPML,para.Rcoef)
    g_eps=0.0
    g_sig=0.0
    rhs=[]
    pool=Pool(processes=8,maxtasksperchild=50)
    res_l=[]
    
    for index,value in enumerate(para.source_site):
        res=pool.apply_async(calculate_gradient,args=(epsilon,sigma,index,CPML_Params,para))
        res_l.append(res)
    pool.close()
    pool.join()

    
    for res in res_l:
        result=res.get()
        rhs.append(result[0])
        g_eps+=result[1]
        g_sig+=result[2]
        del result
    rhs=np.array(rhs)        
    f=0.5*np.linalg.norm(rhs.flatten(),2)**2
#    f_TolVar,g_TolVar=calculate_toltal_variation_model(epsilon,sigma,para.dx,para.dz)
    normagra_epsilon,g_epsilon=cal_toltal_variation(epsilon,para.dx,para.dz)
    normagra_sigma,g_sigma=cal_toltal_variation(sigma,para.dx,para.dz)
    f_epsilon=np.sum(normagra_epsilon.flatten())
    f_sigma=np.sum(normagra_sigma.flatten())
#    g_TolVar=np.hstack((g_rho_TolVar.flatten(),g_vp_TolVar.flatten()))
    lambda_epsilon=lambda_*f/(f_epsilon+f)
    lambda_sigma=lambda_*f/(f_sigma+f)
    f+=lambda_epsilon*f_epsilon+lambda_sigma*f_sigma
#    g_rho[:]=0.0
    g_eps+=lambda_epsilon*g_epsilon.flatten()
    g_sig+=lambda_sigma*g_sigma.flatten()
    g=np.hstack((g_eps,g_sig))
#    print('**********',lambda_,'**********')
    print('Misfit elapsed time is %s seconds !'%str(time.time()-start_time))
    return f,g
