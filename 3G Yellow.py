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
mb2=np.array([215.4510933088003])
sb2=np.array([18.0247770404578])
mg2=np.array([234.0041244762171])
sg2=np.array([6.54973545834207])
mr2=np.array([234.00530400106925])
sr2=np.array([6.5486748652304865])

#215.4510933088003 234.0041244762171 234.00530400106925
#18.0247770404578 6.54973545834207 6.5486748652304865


#234.34650514440594 195.3388390716529 234.37641246485498
#6.762192460722822 33.992273313407004 6.736515620701216


# In[4]:


ans_b2=gaussian(x, mb2, sb2)
ans_g2=gaussian(x, mg2, sg2)
ans_r2=gaussian(x, mr2, sr2)


# In[5]:


plt.plot(ans_b2, 'b')
print(max(ans_b2))
#plt.show()

plt.plot(ans_g2, 'g')
print(max(ans_g2))
#plt.show()


plt.plot(ans_r2, 'r')
print(max(ans_r2))
plt.show()


# In[ ]:





# In[ ]:





# In[9]:


c=cv2.VideoCapture("/home/sanket/Desktop/ENPM673/Untitled Folder/detectbuoy.avi")

while (True):
    ret,image=c.read()
    image_r=image[:,:,2]
    image_g = image[:,:,1]
    image_b=image[:,:,0]
    check=0
    if ret == True:
        img_out2=np.zeros(image_r.shape, dtype = np.uint8)
        check+check+1
        for i in range(0,image_r.shape[0]):
            for j in range(0,image_r.shape[1]):
                y=image_r[i][j]
                z= image_g[i][j]
                if check < 50:
                    value=130
                else:
                    value = 200
                if ((ans_r2[y] +ans_r2[z])/2) > 0.05  and ((ans_b2[y] +ans_b2[z])/2) < 0.015 and image_b[i][j]<value:
                    img_out2[i][j]=255
                else:
                    img_out2[i][j]=0
                    
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
                center = (int(x),int(y)-1)
                radius = int(radius) - 1
                if radius > 12:
                    cv2.circle(image,center,radius,(0,255,255),2)
                    
        cv2.imshow("Threshold",dilation2)
        cv2.imshow('YoYo1', image)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break      # wait for ESC key to exit

    else:
        break
        
c.release()
cv2.destroyAllWindows()   


# In[ ]:





# In[ ]:




