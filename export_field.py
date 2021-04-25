# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:53:15 2020

@author: ssff
"""

import pandas as pd
import numpy as np
import phase_analyze as pa
import vector_map as vm

def obtain_xy_field(para_a, size):
    U = []
    V = []
    for i in range(0,size[0]):
        for j in range(0,size[1]):
            para = para_a[i,j,:]
            x = para[0]
            y = para[1]
            U.append(x)
            V.append(y)
    return [U, V]

def export_3d_field(f, size): #f: 3*(L1*l2)list
    dic = {'A':f[0],'theta':f[1]}
    data = pd.DataFrame(dic)
    data.to_excel('2d_field_2_%d_%d.xlsx'%(size[0],size[1]))

if __name__ == '__main__':
    #obtain x and y field
    s_amp = ['./IP/amp0-01.tif','./IP/amp30-01.tif','./IP/amp60-01.tif','./IP/amp90-01.tif']
    s_phase = ['./IP/pha0-01.tif','./IP/pha30-01.tif','./IP/pha60-01.tif','./IP/pha90-01.tif']
    l_a, l_p = [], []
    for i in s_amp:
        l_a.append(pa.import_image(i))
    size = l_a[0].shape
    n = 1
    for i in s_phase:
        if n != 3:
            pha_1 = pa.import_image(i)
            para = pa.gauss_fit(pha_1, size)[1]
            mid_p = pa.middle_point(para)
            pha_2= pa.generate_new_phase1(pha_1,size,mid_p)
            l_p.append(pha_2)
            n += 1
        else:
            pha_1 = pa.import_image(i)
            para = pa.gauss_fit(pha_1, size)[1]
            mid_p = pa.middle_point(para)
            pha_2= pa.generate_new_phase2(pha_1,size,mid_p)
            l_p.append(pha_2)
            n += 1   
    l_p = vm.generate_new_amp(l_a, l_p)
    a_l = [90, 120, 150, 180]
    a_l = [i*np.pi/180 for i in a_l]
    para_a = vm.traverse_all_point(l_p,a_l,size)
    xy_field = obtain_xy_field(para_a, size)
    
    #obtain z field, if only xy field needed, this 
    '''
    oop_amp = pa.import_image('./OOP/data/OOP_amp-01.tif')
    oop_pha = pa.import_image('./OOP/data/OOP_pha-01.tif')   
    para = pa.gauss_fit(oop_pha, size)[1]
    mid_p = pa.middle_point(para)
    oop_pha_n = pa.generate_new_phase2(oop_pha,size,mid_p)
    z_field = vm.generate_new_amp([oop_amp], [oop_pha_n])[0]
    z_field.resize((1,size[0]*size[1]))
    z_field = z_field.tolist()
    
    field = xy_field + z_field'''
    export_3d_field(xy_field, size)
    
    
    