# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:36:11 2020

@author: ssff
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def gen_new_data(a_l, para):
    one = np.ones(len(a_l))
    phi = np.array(a_l)
    nd = para[0]*np.cos(phi - para[1]*one - np.pi/2*one)
    return nd

def cal_error(l_a, a_l, para_a, size):
    n = len(a_l)
    arr = np.array(l_a)
    R_2 = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            raw_data = arr[:,i,j]
            m = np.mean(raw_data)
            d1 = raw_data - m*np.ones(n)
            S_tot = np.dot(d1, d1)
            new_data = gen_new_data(a_l, para_a[i,j])
            d2 = new_data - m*np.ones(n)
            S_rev = np.dot(d2, d2)
            R_2[i,j] = (np.dot(d1, d2)/np.sqrt(S_tot*S_rev))**2
    return R_2

def plot_dev(l_a, a_l, para_a, size):
    dev_a = cal_error(l_a, a_l, para_a, size)
    #2D
    mpimg.imsave('1.png',dev_a, )
    plt.imshow(dev_a, cmap='spring')
    plt.colorbar()
    plt.show()
    
    '''#3D
    x = np.arange(0,size[0],1)
    y = np.arange(0,size[1],1)
    X, Y = np.meshgrid(x,y)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, dev_a, rstride=1, cstride=1, cmap='rainbow')
    plt.show()'''
    
    
    
    