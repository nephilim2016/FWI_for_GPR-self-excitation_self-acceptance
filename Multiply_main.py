#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 14:57:58 2018

@author: nephilim
"""
from multiprocessing import Pool
import numpy as np
import time
import Add_CPML
import Time_loop
import Create_Model
import Wavelet

if __name__=='__main__':  
    start_time=time.time() 
    xl=101
    zl=101
    dx=5
    dz=5
    k_max=1000
    dt=5e-4
    ricker_freq=30
    t=np.arange(0,k_max)*dt
    f=Wavelet.ricker(t,ricker_freq)
    
    CPML=12
    Npower=2
    k_max_CPML=3
    alpha_max_CPML=ricker_freq*np.pi
    Rcoef=1e-8
    rho,vp=Create_Model.Abnormal_Model(xl,zl,CPML)
    vp_max=max(vp.flatten())
    
    x_site=[CPML+8]*6+[40,60,80,100]+[CPML+8+xl-1]*6+[40,60,80,100]
    z_site=[20,40,60,80,100,110]+[CPML+8+zl-1]*4+[20,40,60,80,100,110]+[CPML+8]*4
    source_site=np.column_stack((x_site,z_site))
    
    ref_pos_x=[CPML+8]*zl+[CPML+8+xl-1]*zl+list(range(CPML+8+1,CPML+8+xl-1))+list(range(CPML+8+1,CPML+8+xl-1))
    ref_pos_z=list(range(CPML+8,CPML+8+zl))+list(range(CPML+8,CPML+8+zl))+[CPML+8]*(xl-2)+[CPML+8+zl-1]*(xl-2)
    ref_pos=np.column_stack((ref_pos_x,ref_pos_z))    
    
    CPML_Params=Add_CPML.Add_CPML(xl,zl,CPML,vp_max,dx,dz,dt,Npower,k_max_CPML,alpha_max_CPML,Rcoef)
 
    result=Time_loop.time_loop(xl,zl,dx,dz,dt,rho,vp,CPML_Params,f,k_max,source_site[10],ref_pos)
    
    pool=Pool(processes=8)
    res_l=[]
    Process_jobs=[]
    for value in source_site:
        res=pool.apply_async(Time_loop.time_loop,(xl,zl,dx,dz,dt,rho,vp,CPML_Params,\
                                                  f,k_max,value,ref_pos))
        res_l.append(res)
    pool.close()
    pool.join()
    for res in res_l:
        result=res.get()
        np.save('%sx_%sz_pdata.npy'%(result[0][0],result[0][1]),result[1])
        np.save('%sx_%sz_record.npy',result[2])
        del result
    print('Elapsed time is %s seconds !'%str(time.time()-start_time))
