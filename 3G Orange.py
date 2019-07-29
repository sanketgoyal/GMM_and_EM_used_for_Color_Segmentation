#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from numpy import linalg as LA
import math
import scipy.stats as stats


# In[2]:


def gaussian(x, mu, sig):
    return ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))


# In[3]:


x=list(range(0, 256))
mb1=np.array([170.1123])
sb1=np.array([36.1331436])
mg1=np.array([239.95395])
sg1=np.array([7.3541856])
mr1=np.array([252.3011604])
sr1=np.array([2.373163])


# In[4]:


ans_b1=gaussian(x, mb1, sb1)
ans_g1=gaussian(x, mg1, sg1)
ans_r1=gaussian(x, mr1, sr1)


# In[5]:


plt.plot(ans_b1, 'b')
print(max(ans_b1))
#plt.show()

plt.plot(ans_g1, 'g')
print(max(ans_g1))
#plt.show()


plt.plot(ans_r1, 'r')
print(max(ans_r1))
plt.show()


# In[6]:


c=cv2.VideoCapture("/home/sanket/Desktop/ENPM673/Untitled Folder/detectbuoy.avi")

while (True):
    ret,image=c.read()
    image_r=image[:,:,2]
    image_b=image[:,:,0]
    if ret == True:
        img_out1=np.zeros(image_r.shape, dtype = np.uint8)
         
        for i in range(0,image_r.shape[0]):
            for j in range(0,image_r.shape[1]):
                y=image_r[i][j]
                
                if ans_r1[y]>0.15 and image_b[i][j]<150:
                    #print(ans_r[y], 'r')
                    img_out1[i][j]=255
                    
                if ans_g1[y]>0.02 and image_b[i][j]<150:
                    #print(ans_g[y], 'g')
                    img_out1[i][j]=0
                
                if ans_b1[y]>0.001 and image_b[i][j]<150:
                    #print(ans_b[y], 'b')
                    img_out1[i][j]=0
                    
                    
                    
        ret, threshold = cv2.threshold(img_out1, 240, 255, cv2.THRESH_BINARY)
        kernel1 = np.ones((2,2),np.uint8)

    
        dilation1 = cv2.dilate(threshold,kernel1,iterations = 6)
        _,contours1,_= cv2.findContours(dilation1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours1:
            if cv2.contourArea(contour) > 20:
                (x,y),radius = cv2.minEnclosingCircle(contour)
                center = (int(x),int(y))
                radius = int(radius)
                if radius > 13:
                    cv2.circle(image,center,radius,(0,0,255),2)
                    
        cv2.imshow("Threshold",dilation1)
        cv2.imshow('YoYo1', image)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break      # wait for ESC key to exit

    else:
        break
        
c.release()
cv2.destroyAllWindows()   


# In[ ]:




