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
    pixel = []
    g1, g2, g3, g4 = [], [], [], []
    y = []
    
    images=[]
    path = "/home/sanket/Desktop/ENPM673/Untitled Folder/buoy pics/Green/"
    for image in os.listdir(path):
        images.append(image)
    
    for image in images:
        image = cv.imread("%s%s"%(path,image))
        image = image[:, :, 1]
        r, c = image.shape
        for j in range(0, r):
            for m in range(0, c):
                im = image[j][m]
                # print(im)
                pixel.append(im)
    print(len(pixel))
    n = 0
    mean1 = 190
    mean2 = 150
    mean3 = 250
    #mean4 = 100
    std1 = 10
    std2 = 10
    std3 = 10
    #std4 = 10
    while (n != 50):
        prob1 = []
        prob2 = []
        prob3 = []
        prob4 = []
        b1 = []
        b2 = []
        b3 = []
        b4 = []
        for im in pixel:
            p1 = calculateProbabilty(im, mean1, std1)
            prob1.append(p1)
            p2 = calculateProbabilty(im, mean2, std2)
            prob2.append(p2)
            p3 = calculateProbabilty(im, mean3, std3)
            prob3.append(p3)

            b1.append((p1 * (1 / 3)) / (p1 * (1 / 3) + p2 * (1 / 3) + p3 * (1 / 3) ))
            b2.append((p2 * (1 / 3)) / (p1 * (1 / 3) + p2 * (1 / 3) + p3 * (1 / 3) ))
            b3.append((p3 * (1 / 3)) / (p1 * (1 / 3) + p2 * (1 / 3) + p3 * (1 / 3) ))
            
        mean1 = np.sum(np.array(b1) * np.array(pixel)) / np.sum(np.array(b1))
        mean2 = np.sum(np.array(b2) * np.array(pixel)) / np.sum(np.array(b2))
        mean3 = np.sum(np.array(b3) * np.array(pixel)) / np.sum(np.array(b3))
        #mean4 = np.sum(np.array(b4) * np.array(pixel)) / np.sum(np.array(b4))
        
        std1 = (np.sum(np.array(b1) * ((np.array(pixel)) - mean1) ** (2)) / np.sum(np.array(b1))) ** (1 / 2)
        std2 = (np.sum(np.array(b2) * ((np.array(pixel)) - mean2) ** (2)) / np.sum(np.array(b2))) ** (1 / 2)
        std3 = (np.sum(np.array(b3) * ((np.array(pixel)) - mean3) ** (2)) / np.sum(np.array(b3))) ** (1 / 2)
        #std4 = (np.sum(np.array(b4) * ((np.array(pixel)) - mean4) ** (2)) / np.sum(np.array(b4))) ** (1 / 2)
        n = n + 1
        print(mean1, mean2, mean3)
        print(std1, std2, std3)
    print('final mean- ',mean1,mean2,mean3)
    print('final strd- ',std1, std2, std3)


# In[ ]:


em_gmm()


# In[ ]:




