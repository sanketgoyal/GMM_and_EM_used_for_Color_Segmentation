#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from numpy import linalg as LA
import math
import scipy.stats as stats
count = 0
from scipy.stats import norm

def gaussian(x, mu, sig):
    return ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))

images=[]
path = "/home/sanket/Desktop/ENPM673/Untitled Folder/buoy pics/Green/"
for image in os.listdir(path):
    images.append(image)
    
hitogram_b=np.zeros((256,1))
hitogram_g=np.zeros((256,1))
hitogram_r=np.zeros((256,1))

for image in images:
    imgr = cv2.imread("%s%s"%(path,image))
    
    #imgr = img[2,:,:]
    img = cv2.GaussianBlur(imgr,(5,5),0)
    #print(np.shape(img))
    color = ("b", "g", "r") 
    for i,col in enumerate(color):
        if col =='b':
            histr_b = cv2.calcHist([img],[i],None,[256],[0,256])
            #print(histr_b.shape)
            hitogram_b=np.column_stack((hitogram_b,histr_b))
            
            #print(histr)
            #plt.plot(histr_b,color = col)
            #plt.xlim([0,256])
        if col =='g':
            histr_g = cv2.calcHist([img],[i],None,[256],[0,256])
            #print(histr_g.shape)
            hitogram_g=np.column_stack((hitogram_g,histr_g))
            #print(histr)
            #plt.plot(histr_g,color = col)
            #plt.xlim([0,256])
        if col =='r':
            histr_r = cv2.calcHist([img],[i],None,[256],[0,256])
            #print(histr_r.shape)
            hitogram_r=np.column_stack((hitogram_r,histr_r))
            #print(histr)
            #plt.plot(histr_r,color = col)
            #plt.xlim([0,256])

histogram_avg_r = np.sum(hitogram_r, axis=1) / (hitogram_r.shape[1]-1)
histogram_avg_g = np.sum(hitogram_g, axis=1) / (hitogram_g.shape[1]-1)
histogram_avg_b = np.sum(hitogram_b, axis=1) / (hitogram_b.shape[1]-1)
plt.plot(histogram_avg_r,color = 'r')

plt.plot(histogram_avg_g,color = 'g')
plt.plot(histogram_avg_b,color = 'b')
plt.show()

x=np.array((range(0,256))).T
(mean, stds) = cv2.meanStdDev(img)
#print(mean)
#print(stds)
y= np.array
x=list(range(0, 255))



b_mean = mean[0]
b_std = stds[0]

g_mean = mean[1]
g_std = stds[1]

r_mean = mean[2]
r_std = stds[2]


ans_b=gaussian(x, b_mean, b_std)
ans_g=gaussian(x, g_mean, g_std)
ans_r=gaussian(x, r_mean, r_std)

#plt.plot(ans_b)
#plt.show()


plt.plot(ans_g)
plt.show()


#plt.plot(ans_r)
#plt.show()

     
        


# In[ ]:




