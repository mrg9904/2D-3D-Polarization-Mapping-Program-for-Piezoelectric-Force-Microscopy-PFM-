# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:33:33 2020

@author: ssff
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import leastsq
import phase_analyze as pa
import error_analyze as ea

def generate_new_amp(l_a,l_p):
    pair_l = list(zip(l_a,l_p))
    amp_n = []
    for i in pair_l:
        amp_n.append(i[0]*i[1])
    return amp_n

def fun(p,phi):
    A, theta = p
    return A*np.cos(phi - theta - np.pi/2)

def error(p,x,y):
    return fun(p,x) - y

def regressive_alz(l,p0):
    #l contains 2 lists of angle and amplitude
    para =leastsq(error, p0, args=(l[0],l[1]))  
    return para[0]

def traverse_all_point(l_a,a_l,size):
    arr = np.array(l_a)
    p0 = np.mean(arr[0]), 0
    L1 = size[0]
    L2 = size[1]
    p_l = []
    for i in range(L1):
        for j in range(L2):
             p_l.append(arr[:,i,j])
    para_l = []
    n = 0
    for i in p_l:
        para = [a_l, i]
        para_s = regressive_alz(para,p0)
        if para_s[0] < 0:
            para_s[0] *= -1
            para_s[1] += np.pi
        while(para_s[1] < 0):
            para_s[1] += 2*np.pi
        while(para_s[1] > 2*np.pi):
            para_s[1] -= 2*np.pi
        para_l.append(para_s)
        n += 1
        if n % L1 == 0:
            print(n/L1)
    para_a = np.array(para_l)
    para_a.resize((L1,L2,2))
    return para_a

def plot_vector_map(para_a,size):
    U = []
    V = []
    for i in range(0,size[0],2):
        for j in range(0,size[1],2):
            para = para_a[i,j,:]
            x = para[0]*np.cos(para[1])
            y = para[0]*np.sin(para[1])
            U.append(x)
            V.append(y)
    U = np.array(U)
    V = np.array(V)
    L1 = size[0]//2+1
    L2 = size[1]//2+1
    U.resize((L1,L2))
    V.resize((L1,L2))
    U_verse = np.zeros((L1,L2))
    V_verse = np.zeros((L1,L2))
    for i in range(L1):
        U_verse[i] = U[L1-1-i]
        V_verse[i] = V[L1-1-i]    
    fig = plt.figure(figsize=(size[1]//5,size[0]//5))
    plt.quiver(U_verse,V_verse,headwidth=2,headlength=3,minshaft=3,scale=5000,scale_units='width')
    #plt.quiver(U_verse,V_verse,minshaft=3)
    fig.savefig('1.png')
    
def trigono_demo(l,para_a,a_l,size,demo_ind):
    l = np.array(l)
    L1 = size[0]
    L2 = size[1]
    print(L1,L2)
    #regressive curve -- curve
    demo_para = para_a[demo_ind[0]][demo_ind[1]]
    phi = np.linspace(np.pi/4,np.pi*5/4,50)
    print(demo_para)
    theta = np.ones(50)*demo_para[1]
    P = demo_para[0]*np.cos(phi - theta - np.pi/2)
    #raw data -- point
    p_demo_l = l[:,demo_ind[0],demo_ind[1]]
    #plot
    fig = plt.figure()
    plt.plot(phi*180/np.pi,P,'-r')
    plt.plot([x*180/np.pi for x in a_l], p_demo_l,'ob')
    plt.xlabel('sample rotation angle/°')
    plt.ylabel('projected polarization amplitude/(a.u.)')
    plt.yticks([])
    plt.show()
    #fig.savefig('fit_plot_201009/6.png',dpi=600)
    dic = {}
    #dic['point_x'] = np.array([x*180/np.pi for x in a_l])
    #dic['point_y'] = p_demo_l
    dic['curve_x'] = (phi*180/np.pi)
    dic['curve_y'] = P
    df = pd.DataFrame(dic)
    df.to_excel('fitting_data_curve_66_38.xlsx')

    
if __name__ == '__main__':
    s_amp = ['./data/amp0-01.tif','./data/amp30-01.tif','./data/amp60-01.tif','./data/amp90-01.tif']
    s_phase = ['./data/pha0-01.tif','./data/pha30-01.tif','./data/pha60-01.tif','./data/pha90-01.tif']
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
    l_p = generate_new_amp(l_a, l_p)
    a_l = [90, 120, 150, 180]
    a_l = [i*np.pi/180 for i in a_l]
    para_a = traverse_all_point(l_p,a_l,size)
    '''for i in range(1,10):
        print(i)
        #[70,71],[38,39],[68,78],[1,21],[61,71]'''
    demo_ind = [66,38]
    trigono_demo(l_p,para_a,a_l,size,demo_ind)
    #plot_vector_map(para_a, size)
    #ea.plot_dev(l_p, a_l, para_a, size)

    
    
    