#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
from numpy import linalg as LA
import math
import scipy.stats as stats


def calculateGaussianEquation(x_co, mean, std):
    return (1 / (std * math.sqrt(2 * math.pi))) * np.exp(-np.power(x_co - mean, 2.) / (2 * np.power(std, 2.)))


def calculateProbabilty(x_co, mean, std):
    return (1 / (std * math.sqrt(2 * math.pi))) * (math.exp(-((x_co - mean) ** 2) / (2 * (std) ** 2)))


def em_gmm():
    pixel1 = []
    pixel2=[]
    
    images=[]
    path = "/home/sanket/Desktop/ENPM673/Untitled Folder/buoy pics/Yellow/"
    for image in os.listdir(path):
        images.append(image)
    
    for image in images:
        image = cv.imread("%s%s"%(path,image))
    
    #for i in range(1, 140+ 1):
        #image = cv.imread("D:/MD/2nd sem/perception/pro3/yellow_buoy/" + str(i) + ".jpg")
        #image=cv.cvtColor(image,cv.COLOR_BGR2HSV)
        
        image1 = (image[:,:,1])
        r, c= image1.shape
        for j in range(0,r):
            for m in range(0, c):
                im1 = image1[j][m]
                pixel1.append(im1)
        image2 = (image[:, :, 2])
        r, c = image2.shape
        for j in range(0, r):
            for m in range(0, c):
                im2 = image2[j][m]
                pixel2.append(im2)
    #print(len(pixel1), len(pixel2))
    n = 0
    mean1 = 100
    mean2 = 0
    mean3 = 250
    std1 = 10
    std2 = 10
    std3 = 10
    while(n!=50):
        b11 = []
        b21 = []
        b31 = []
        b12=[]
        b22=[]
        b32=[]
        for im1 in pixel1:
            p11 = calculateProbabilty(im1, mean1, std1)
            p21 = calculateProbabilty(im1, mean2, std2)
            p31 = calculateProbabilty(im1, mean3, std3)
            b11.append((p11*(1/3))/(p11*(1/3) + p21*(1/3) + p31*(1/3)))
            b21.append((p21 * (1 / 3)) / (p11 * (1 / 3) + p21 * (1 / 3) + p31 * (1 / 3)))
            b31.append((p31 * (1 / 3)) / (p11 * (1 / 3) + p21 * (1 / 3) + p31 * (1 / 3)))
        for im2 in pixel2:
            p12 = calculateProbabilty(im2, mean1, std1)
            p22 = calculateProbabilty(im2, mean2, std2)
            p32 = calculateProbabilty(im2, mean3, std3)
            b12.append((p12*(1/3))/(p12*(1/3) + p22*(1/3) + p32*(1/3)))
            b22.append((p22 * (1 / 3)) / (p12 * (1 / 3) + p22 * (1 / 3) + p32 * (1 / 3)))
            b32.append((p32 * (1 / 3)) / (p12 * (1 / 3) + p22 * (1 / 3) + p32 * (1 / 3)))
        m11 = np.sum(np.array(b11)* np.array(pixel1))/np.sum(np.array(b11))
        m21 = np.sum(np.array(b21) * np.array(pixel1)) / np.sum(np.array(b21))
        m31 = np.sum(np.array(b31) * np.array(pixel1)) / np.sum(np.array(b31))
        s11 = (np.sum(np.array(b11)*((np.array(pixel1)) - mean1)**(2))/np.sum(np.array(b11)))**(1/2)
        s21 = (np.sum(np.array(b21) * ((np.array(pixel1)) - mean2) ** (2)) / np.sum(np.array(b21))) ** (1 / 2)
        s31 = (np.sum(np.array(b31) * ((np.array(pixel1)) - mean3) ** (2)) / np.sum(np.array(b31))) ** (1 / 2)
        m12 = np.sum(np.array(b12)* np.array(pixel2))/np.sum(np.array(b12))
        m22 = np.sum(np.array(b22) * np.array(pixel2)) / np.sum(np.array(b22))
        m32 = np.sum(np.array(b32) * np.array(pixel2)) / np.sum(np.array(b32))
        s12 = (np.sum(np.array(b12)*((np.array(pixel2)) - mean1)**(2))/np.sum(np.array(b12)))**(1/2)
        s22 = (np.sum(np.array(b22) * ((np.array(pixel2)) - mean2) ** (2)) / np.sum(np.array(b22))) ** (1 / 2)
        s32 = (np.sum(np.array(b32) * ((np.array(pixel2)) - mean3) ** (2)) / np.sum(np.array(b32))) ** (1 / 2)
        n = n + 1
        mean1=(m11+m12)/2
        mean2 = (m21 + m22) / 2
        mean3 = (m31 + m32) / 2
        std1 = (s11 + s12) / 2
        std2 = (s21 + s22) / 2
        std3 = (s31 + s32) / 2
        print(mean1, mean2, mean3)
        print(std1, std2, std3)


# In[ ]:


em_gmm()


# In[ ]:




