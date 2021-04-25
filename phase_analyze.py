# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 10:06:29 2020

@author: ssff
"""

from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import numpy as np

def import_image(s):
    pic = mpimg.imread(s)
    pic = np.dot(pic[...,:3], [0.299, 0.587, 0.114])
    return pic
    
def gauss_fit(pic,size):
    samples = pic.reshape([size[0]*size[1],1])
    clf = mixture.GaussianMixture(n_components=3, covariance_type='full',tol=1e-10, max_iter=100000)
    clf.fit(samples)
    m1, m2, m3 = clf.means_
    w1, w2, w3 = clf.weights_
    c1, c2, c3 = clf.covariances_
    para = [[m1, m2, m3],[w1, w2, w3],[c1, c2, c3]]
    return samples, para

def plot_gauss(yourdata, para):
    histdist = plt.hist(yourdata, 100, normed=True)
    [[m1, m2, m3],[w1, w2, w3],[c1, c2, c3]] = para
    plotgauss1 = lambda x: plt.plot(x,w1*scipy.stats.norm.pdf(x,m1,np.sqrt(c1))[0], linewidth=3)
    plotgauss2 = lambda x: plt.plot(x,w2*scipy.stats.norm.pdf(x,m2,np.sqrt(c2))[0], linewidth=3)
    plotgauss3 = lambda x: plt.plot(x,w3*scipy.stats.norm.pdf(x,m3,np.sqrt(c3))[0], linewidth=3)
    plotgauss1(histdist[1])
    plotgauss2(histdist[1])
    plotgauss3(histdist[1])
    
def middle_point(para):
    m_l = para[0]
    c_l = para[2]
    pair_l = list(zip(m_l,c_l))
    pair_l = sorted(pair_l,key = lambda x:x[0])
    left_p = pair_l[0][0] + 3*np.sqrt(pair_l[0][1])
    right_p = pair_l[2][0] - 3*np.sqrt(pair_l[2][1])
    m_p = (left_p + right_p) / 2
    return m_p

def generate_new_phase1(pic,size,m_p):
    pic_n = pic
    for i in range(size[0]):
        for j in range(size[1]):
            if pic[i,j] < m_p:
                pic_n[i,j] = -1
            else:
                pic_n[i,j] = 1
    return pic_n

def generate_new_phase2(pic,size,m_p):
    pic_n = pic
    for i in range(size[0]):
        for j in range(size[1]):
            if pic[i,j] < m_p:
                pic_n[i,j] = 1
            else:
                pic_n[i,j] = -1
    return pic_n

def plot_new_phase(pic, size):
    pic_n = pic
    for i in range(size[0]):
        for j in range(size[1]):
            if pic[i,j] == 1:
                pic_n[i,j] = 0
            else:
                pic_n[i,j] = 255
    plt.imshow(pic_n)
                
if __name__ == '__main__':
    pic = import_image('./data/amp0-01.tif')
    '''size = pic.shape
    samples, para = gauss_fit(pic, size)
    plot_gauss(samples, para)
    m_p = middle_point(para)
    plot_new_phase(generate_new_phase(pic,size,m_p),size)'''
    pic_seg = pic[150,0]
    print(pic_seg)

'''
1. 注意在主函数中实现功能循环，而不是在分函数中
分函数应该是实现单一功能的最简函数。
'''




