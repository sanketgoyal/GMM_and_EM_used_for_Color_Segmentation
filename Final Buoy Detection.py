#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from numpy import linalg as LA
import math
import scipy.stats as stats


# In[3]:


def gaussian(x, mu, sig):
    return ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))


# In[4]:


x=list(range(0, 256))
mb1=np.array([170.1123])
sb1=np.array([36.1331436])
mg1=np.array([239.95395])
sg1=np.array([7.3541856])
mr1=np.array([252.3011604])
sr1=np.array([2.373163])


# In[5]:


x=list(range(0, 256))
mb2=np.array([215.4510933088003])
sb2=np.array([18.0247770404578])
mg2=np.array([234.0041244762171])
sg2=np.array([6.54973545834207])
mr2=np.array([234.00530400106925])
sr2=np.array([6.5486748652304865])


# In[6]:


x=list(range(0, 256))
mb3=np.array([230.92855834531522])
sb3=np.array([11.825125082753653])
mg3=np.array([200.98927408174694])
sg3=np.array([12.952317147474373])
mr3=np.array([244.4299727613101])
sr3=np.array([5.067551652277579])


# In[7]:


ans_b1=gaussian(x, mb1, sb1)
ans_g1=gaussian(x, mg1, sg1)
ans_r1=gaussian(x, mr1, sr1)


# In[8]:


ans_b2=gaussian(x, mb2, sb2)
ans_g2=gaussian(x, mg2, sg2)
ans_r2=gaussian(x, mr2, sr2)


# In[9]:


ans_b3=gaussian(x, mb3, sb3)
ans_g3=gaussian(x, mg3, sg3)
ans_r3=gaussian(x, mr3, sr3)


# In[10]:


plt.plot(ans_b1, 'b')
print(max(ans_b1))
#plt.show()

plt.plot(ans_g1, 'g')
print(max(ans_g1))
#plt.show()


plt.plot(ans_r1, 'r')
print(max(ans_r1))
plt.show()


# In[11]:


plt.plot(ans_b2, 'b')
print(max(ans_b2))
#plt.show()

plt.plot(ans_g2, 'g')
print(max(ans_g2))
#plt.show()


plt.plot(ans_r2, 'r')
print(max(ans_r2))
plt.show()


# In[12]:


plt.plot(ans_b3, 'b')
print(max(ans_b3))
#plt.show()

plt.plot(ans_g3, 'g')
print(max(ans_g3))
#plt.show()


plt.plot(ans_r3, 'r')
print(max(ans_r3))
plt.show()


# In[15]:


c=cv2.VideoCapture("/home/sanket/Desktop/ENPM673/Untitled Folder/detectbuoy.avi")

while (True):
    ret,image=c.read()
    image_r=image[:,:,2]
    image_g = image[:,:,1]
    image_b=image[:,:,0]
    check = 0
    
    
    if ret == True:
        img_out1=np.zeros(image_r.shape, dtype = np.uint8)
        img_out2=np.zeros(image_r.shape, dtype = np.uint8)
        img_out3=np.zeros(image_g.shape, dtype = np.uint8)
        check+check+1
        for i in range(0,image_r.shape[0]):
            for j in range(0,image_r.shape[1]):
                y=image_r[i][j]
                z=image_g[i][j]
                
                if ans_r1[y]>0.15 and image_b[i][j]<150:
                    #print(ans_r[y], 'r')
                    img_out1[i][j]=255
                    
                if ans_g1[y]>0.02 and image_b[i][j]<150:
                    #print(ans_g[y], 'g')
                    img_out1[i][j]=0
                
                if ans_b1[y]>0.001 and image_b[i][j]<150:
                    #print(ans_b[y], 'b')
                    img_out1[i][j]=0
                    
                if ans_r3[z]>0.06 and ans_g3[z]<0.02 and ans_b3[z]<0.02 and image_r[i][j]<200:
                    img_out3[i][j]=255
                else:
                    img_out3[i][j]=0 
                    
                if check < 50:
                    value=130
                else:
                    value = 200
                if ((ans_r2[y] +ans_r2[z])/2) > 0.05  and ((ans_b2[y] +ans_b2[z])/2) < 0.015 and image_b[i][j]<value:
                    img_out2[i][j]=255
                else:
                    img_out2[i][j]=0
                    
        ret, threshold1 = cv2.threshold(img_out1, 240, 255, cv2.THRESH_BINARY)
        kernel1 = np.ones((2,2),np.uint8)
        dilation1 = cv2.dilate(threshold1,kernel1,iterations = 6)
        _,contours1,_= cv2.findContours(dilation1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours1:
            if cv2.contourArea(contour) > 20:
                (x,y),radius = cv2.minEnclosingCircle(contour)
                center = (int(x),int(y))
                radius = int(radius)
                if radius > 13:
                    cv2.circle(image,center,radius,(0,0,255),2)
                    
                    
        ret, threshold3 = cv2.threshold(img_out3, 240, 255, cv2.THRESH_BINARY)
        kernel3 = np.ones((2,2),np.uint8)
        dilation3 = cv2.dilate(threshold3,kernel3,iterations =9)
        _,contours3, _= cv2.findContours(dilation3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours3:
            if cv2.contourArea(contour) >  30:
                (x,y),radius = cv2.minEnclosingCircle(contour)
                center = (int(x),int(y))
                radius = int(radius)
                if radius > 13 and radius < 15.5:
                    cv2.circle(image,center,radius,(0,255,0),2)
                
                
        ret, threshold2 = cv2.threshold(img_out2, 240, 255, cv2.THRESH_BINARY)
        kernel2 = np.ones((3,3),np.uint8)
    
        dilation2 = cv2.dilate(threshold2,kernel2,iterations = 6)
        #dilation=cv2.GaussianBlur(dilation,(5,5),0)
        _,contours2, _= cv2.findContours(dilation2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
        #print("h)
        for contour in contours2:
            #print(contour)
            if cv2.contourArea(contour) >  20:
                (x,y),radius = cv2.minEnclosingCircle(contour)
                center = (int(x),int(y))
                radius = int(radius)
                if radius > 12:
                    cv2.circle(image,center,radius,(0,255,255),2)
                    
        #cv2.imshow("Threshold",dilation1)
        cv2.imshow('YoYo1', image)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break      # wait for ESC key to exit

    else:
        break
        
c.release()
cv2.destroyAllWindows()   


# In[ ]:




