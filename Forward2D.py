#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 14:57:58 2018

@author: nephilim
"""
from multiprocessing import Pool
import numpy as np
import Add_CPML
import Time_loop
import Wavelet
import shutil
import os
from numba import jit

#Forward modelling ------ update_vw
@jit(nopython=True)            
def update_H(xl,zl,dx,dz,dt,epsilon,sigma,npml,a_x,a_z,b_x,b_z,k_x,k_z,Hz,Hx,Ey,memory_dEy_dx,memory_dEy_dz):
#    c1=1.23404
#    c2=-0.10665
#    c3=0.0230364
#    c4=-0.00534239
#    c5=0.00107727
#    c6=-1.6642e-4
#    c7=1.7022e-5
#    c8=-8.5235e-7
    c1 = 1
    mu = 1.2566370614359173e-06
    x_len=xl+0+npml*2
    z_len=zl+0+npml*2
    """ 
    mu*dHz/dt = dEy/dx
    Hz --> v
    Ey --> p
    """
    for j in range(1,z_len-1):
        for i in range(1,x_len-1):
#            value_dp_dx=(c1*(p[i+1][j]-p[i-0][j])+c2*(p[i+2][j]-p[i-1][j])+\
#                             c3*(p[i+3][j]-p[i-2][j])+c4*(p[i+4][j]-p[i-3][j])+\
#                             c5*(p[i+5][j]-p[i-4][j])+c6*(p[i+6][j]-p[i-5][j])+\
#                             c7*(p[i+7][j]-p[i-6][j])+c8*(p[i+8][j]-p[i-7][j]))/dx
            value_dEy_dx=c1*(Ey[i+1][j]-Ey[i-0][j])/dx
                         
            if (i>=npml+0) and (i<x_len-npml-0):
                Hz[i][j]+=value_dEy_dx*dt/mu
                
            elif i<npml+0:
                memory_dEy_dx[i][j]=b_x[i]*memory_dEy_dx[i][j]+a_x[i]*value_dEy_dx
                value_dEy_dx=value_dEy_dx/k_x[i]+memory_dEy_dx[i][j]
                Hz[i][j]+=value_dEy_dx*dt/mu
                
            elif i>=xl-npml-0:
                memory_dEy_dx[i-xl][j]=b_x[i]*memory_dEy_dx[i-xl][j]+a_x[i]*value_dEy_dx
                value_dEy_dx=value_dEy_dx/k_x[i]+memory_dEy_dx[i-xl][j]
                Hz[i][j]+=value_dEy_dx*dt/mu
                
###############################################################################
    """ 
    mu*dHx/dt = - dEy/dz
    Hx --> w
    Ey --> p
    """          
    for j in range(1,z_len-1):
        for i in range(1,x_len-1):
#            value_dp_dz=(c1*(p[i][j+1]-p[i][j-0])+c2*(p[i][j+2]-p[i][j-1])+\
#                         c3*(p[i][j+3]-p[i][j-2])+c4*(p[i][j+4]-p[i][j-3])+\
#                         c5*(p[i][j+5]-p[i][j-4])+c6*(p[i][j+6]-p[i][j-5])+\
#                         c7*(p[i][j+7]-p[i][j-6])+c8*(p[i][j+8]-p[i][j-7]))/dz
            value_dEy_dz=c1*(Ey[i][j+1]-Ey[i][j-0])/dz
                         
            if (j>=npml+0) and (j<z_len-npml-0):
                Hx[i][j]-=value_dEy_dz*dt/mu
                
            elif j<npml+0:
                memory_dEy_dz[i][j]=b_z[j]*memory_dEy_dz[i][j]+a_z[j]*value_dEy_dz
                value_dEy_dz=value_dEy_dz/k_z[j]+memory_dEy_dz[i][j]
                Hx[i][j]-=value_dEy_dz*dt/mu
                
            elif j>=z_len-npml-0:
                memory_dEy_dz[i][j-zl]=b_z[j]*memory_dEy_dz[i][j-zl]+a_z[j]*value_dEy_dz
                value_dEy_dz=value_dEy_dz/k_z[j]+memory_dEy_dz[i][j-zl]
                Hx[i][j]-=value_dEy_dz*dt/mu 

    return Hz,Hx

#Forward modelling ------ update_p
@jit(nopython=True)            
def update_E(xl,zl,dx,dz,dt,ca,cb,npml,a_x,a_z,b_x,b_z,k_x,k_z,Hz,Hx,Ey,memory_dHz_dx,memory_dHx_dz):
#    c1=1.23404
#    c2=-0.10665
#    c3=0.0230364
#    c4=-0.00534239
#    c5=0.00107727
#    c6=-1.6642e-4
#    c7=1.7022e-5
#    c8=-8.5235e-7
    """ 
    epsilon*dEy/dt + sigma*Ey = dHz/dx - dHx/dz
    Hz --> v
    Hx --> w
    Ey --> p
    """
    c1 = 1

    
    x_len=xl+npml*2
    z_len=zl+npml*2
    for j in range(1,z_len-1):
        for i in range(1,x_len-1):
#            value_dv_dx=(c1*(v[i+0][j]-v[i-1][j])+c2*(v[i+1][j]-v[i-2][j])+\
#                         c3*(v[i+2][j]-v[i-3][j])+c4*(v[i+3][j]-v[i-4][j])+\
#                         c5*(v[i+4][j]-v[i-5][j])+c6*(v[i+5][j]-v[i-6][j])+\
#                         c7*(v[i+6][j]-v[i-7][j])+c8*(v[i+7][j]-v[i-8][j]))/dx
#         
#            value_dw_dz=(c1*(w[i][j+0]-w[i][j-1])+c2*(w[i][j+1]-w[i][j-2])+\
#                         c3*(w[i][j+2]-w[i][j-3])+c4*(w[i][j+3]-w[i][j-4])+\
#                         c5*(w[i][j+4]-w[i][j-5])+c6*(w[i][j+5]-w[i][j-6])+\
#                         c7*(w[i][j+6]-w[i][j-7])+c8*(w[i][j+7]-w[i][j-8]))/dz
            value_dv_dx=c1*(Hz[i+0][j]-Hz[i-1][j])/dx
         
            value_dw_dz=c1*(Hx[i][j+0]-Hx[i][j-1])/dz                        

            if (i>=npml+0) and (i<x_len-npml-0) and (j>=npml+0) and (j<z_len-npml-0):
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
                
            elif (i<npml+0) and (j>=npml+0) and (j<z_len-npml-0):
                memory_dHz_dx[i][j]=b_x[i]*memory_dHz_dx[i][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dHz_dx[i][j]
#                p[i][j]+=k_value[i][j]*(value_dv_dx-value_dw_dz)*dt
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
                
            elif (i>=x_len-npml-0) and (j>=npml+0) and (j<z_len-npml-0):
                memory_dHz_dx[i-xl][j]=b_x[i]*memory_dHz_dx[i-xl][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dHz_dx[i-xl][j]
#                p[i][j]+=k_value[i][j]*(value_dv_dx-value_dw_dz)*dt
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
                
            elif (j<npml+0) and (i>=npml+0) and (i<x_len-npml-0):
                memory_dHx_dz[i][j]=b_z[j]*memory_dHx_dz[i][j]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dHx_dz[i][j]
#                p[i][j]+=k_value[i][j]*(value_dv_dx-value_dw_dz)*dt
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
                
            elif (j>=z_len-npml-0) and (i>=npml+0) and (i<x_len-npml-0):
                memory_dHx_dz[i][j-zl]=b_z[j]*memory_dHx_dz[i][j-zl]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dHx_dz[i][j-zl]
#                p[i][j]+=k_value[i][j]*(value_dv_dx-value_dw_dz)*dt
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
                
            elif (i<npml+0) and (j<npml+0):
                memory_dHz_dx[i][j]=b_x[i]*memory_dHz_dx[i][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dHz_dx[i][j]
                
                memory_dHx_dz[i][j]=b_z[j]*memory_dHx_dz[i][j]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dHx_dz[i][j]
                
#                p[i][j]+=k_value[i][j]*(value_dv_dx-value_dw_dz)*dt
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
                
            elif (i<npml+0) and (j>=z_len-npml-0):
                memory_dHz_dx[i][j]=b_x[i]*memory_dHz_dx[i][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dHz_dx[i][j]
                
                memory_dHx_dz[i][j-zl]=b_z[j]*memory_dHx_dz[i][j-zl]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dHx_dz[i][j-zl]
                
#                p[i][j]+=k_value[i][j]*(value_dv_dx-value_dw_dz)*dt
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
                
            elif (i>=x_len-npml-0) and (j<npml+0):
                memory_dHz_dx[i-xl][j]=b_x[i]*memory_dHz_dx[i-xl][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dHz_dx[i-xl][j]
                
                memory_dHx_dz[i][j]=b_z[j]*memory_dHx_dz[i][j]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dHx_dz[i][j]
                
#                p[i][j]+=k_value[i][j]*(value_dv_dx-value_dw_dz)*dt
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
                
            elif (i>=x_len-npml-0) and (j>=z_len-npml-0):
                memory_dHz_dx[i-xl][j]=b_x[i]*memory_dHz_dx[i-xl][j]+a_x[i]*value_dv_dx
                value_dv_dx=value_dv_dx/k_x[i]+memory_dHz_dx[i-xl][j]
                
                memory_dHx_dz[i][j-zl]=b_z[j]*memory_dHx_dz[i][j-zl]+a_z[j]*value_dw_dz
                value_dw_dz=value_dw_dz/k_z[j]+memory_dHx_dz[i][j-zl]
                
#                p[i][j]+=k_value[i][j]*(value_dv_dx-value_dw_dz)*dt
                Ey[i][j]=ca[i][j]*Ey[i][j]+cb[i][j]*(value_dv_dx-value_dw_dz)*dt
    return Ey

#Forward modelling ------ timeloop
def time_loop(xl,zl,dx,dz,dt,epsilon,sigma,CPML_Params,f,k_max,source_site,ref_pos):
#    ep0 = 8.841941282883074e-12
#    epsilon = epsilon*ep0
    npml=CPML_Params.npml        
    Ey=np.zeros((xl+2*npml,zl+2*npml))
    Hz=np.zeros((xl+2*npml,zl+2*npml))
    Hx=np.zeros((xl+2*npml,zl+2*npml))
        
    memory_dEy_dx=np.zeros((2*npml,zl+2*npml))
    memory_dEy_dz=np.zeros((xl+2*npml,2*npml))
    memory_dHz_dx=np.zeros((2*npml,zl+2*npml))
    memory_dHx_dz=np.zeros((xl+2*npml,2*npml))
    
    a_x=CPML_Params.a_x
    b_x=CPML_Params.b_x
    k_x=CPML_Params.k_x
    a_z=CPML_Params.a_z
    b_z=CPML_Params.b_z
    k_z=CPML_Params.k_z
    a_x_half=CPML_Params.a_x_half
    b_x_half=CPML_Params.b_x_half
    k_x_half=CPML_Params.k_x_half
    a_z_half=CPML_Params.a_z_half
    b_z_half=CPML_Params.b_z_half
    k_z_half=CPML_Params.k_z_half
    ca = CPML_Params.ca
    cb = CPML_Params.cb
    
    record=np.zeros((k_max))
    Ey_data=[]  
        
    for tt in range(k_max):
        Hz,Hx=update_H(xl,zl,dx,dz,dt,ca,cb,npml,a_x_half,a_z_half,b_x_half,b_z_half,k_x_half,k_z_half,Hz,Hx,Ey,memory_dEy_dx,memory_dEy_dz)
        Ey=update_E(xl,zl,dx,dz,dt,ca,cb,npml,a_x,a_z,b_x,b_z,k_x,k_z,Hz,Hx,Ey,memory_dHz_dx,memory_dHx_dz)
        Ey[source_site[0]][source_site[1]]+=-cb[source_site[0]][source_site[1]]*f[tt]*dt/dx/dz
        record[tt]=Ey[ref_pos[0],ref_pos[1]]
        Ey_data.append(Ey.copy())
    return source_site,np.array(record)

def Forward_2D(epsilon,sigma,para):
    if not os.path.exists('./forward_data_file'):
        os.makedirs('./forward_data_file')
    else:
        shutil.rmtree('./forward_data_file')
        os.makedirs('./forward_data_file')
        
    t=np.arange(para.k_max)*para.dt
    f=Wavelet.ricker(t,para.ricker_freq)
#    vp_max=max(vp.flatten())
    sigma = para.beta*sigma/para.eta0
    CPML_Params=Add_CPML.Add_CPML(para.xl,para.zl,para.CPML,epsilon,sigma,para.dx,para.dz,para.dt,\
                                  para.Npower,para.k_max_CPML,para.alpha_max_CPML,para.Rcoef)
     
    pool=Pool(processes=8)
    res_l=[]
    for idx in range(len(para.source_site)):
        res=pool.apply_async(time_loop,(para.xl,para.zl,para.dx,para.dz,para.dt,\
                                        epsilon,sigma,CPML_Params,f,para.k_max,\
                                        para.source_site[idx],para.ref_pos[idx]))
        res_l.append(res)
        del res


    profile=np.zeros((para.k_max,len(para.source_site)))
    idx_=0
    for res in res_l:
        result=res.get()
        profile[:,idx_]=result[1]
        idx_+=1
        del result
    np.save('./forward_data_file/Forward_data.npy',profile)
    del res_l
    del profile
    pool.close()
    pool.join()

