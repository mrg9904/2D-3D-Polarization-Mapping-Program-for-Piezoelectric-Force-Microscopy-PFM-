from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def count(t,l):
    m = t[0]
    c = t[1]
    b = [m-3*np.sqrt(c),m+3*np.sqrt(c)]
    co = 0
    for i in l:
        if i > b[0] and i < b[1]:
            co += 1
    return co
            
lena = mpimg.imread('./data/pha0-01.tif')
gray = rgb2gray(lena)
l1 = gray.shape[0]
l2 = gray.shape[1]
yourdata = gray.reshape([l1*l2,1])

clf = mixture.GaussianMixture(n_components=3, covariance_type='full',tol=1e-10, max_iter=100000)
clf.fit(yourdata)
m1, m2, m3 = clf.means_
w1, w2, w3 = clf.weights_
c1, c2, c3 = clf.covariances_
histdist = plt.hist(yourdata, 100, normed=True)
plotgauss1 = lambda x: plt.plot(x,w1*scipy.stats.norm.pdf(x,m1,np.sqrt(c1))[0], linewidth=3)
plotgauss2 = lambda x: plt.plot(x,w2*scipy.stats.norm.pdf(x,m2,np.sqrt(c2))[0], linewidth=3)
plotgauss3 = lambda x: plt.plot(x,w3*scipy.stats.norm.pdf(x,m3,np.sqrt(c3))[0], linewidth=3)
plotgauss1(histdist[1])
plotgauss2(histdist[1])
plotgauss3(histdist[1])

s = [(m1,c1),(m2,c2),(m3,c3)]
s = sorted(s,key = lambda x:x[0])
l = gray.reshape(l1*l2)
co1 = count(s[0],l)
co3 = count(s[2],l)
co2 = l1*l2 - co1 - co3
su = co1 + co2 + co3
#print(co1/su, co2/su, 1-co1/su-co2/su)