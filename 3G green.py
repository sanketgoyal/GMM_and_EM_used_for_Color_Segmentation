#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from numpy import linalg as LA
import math
import scipy.stats as stats


# In[11]:


def gaussian(x, mu, sig):
    return ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))


# In[12]:


x=list(range(0, 256))
mb3=np.array([230.92855834531522])
sb3=np.array([11.825125082753653])
mg3=np.array([200.98927408174694])
sg3=np.array([12.952317147474373])
mr3=np.array([244.4299727613101])
sr3=np.array([5.067551652277579])
#mk=np.array([198.13008990933469])
#sk=np.array([12.533355538527525])

#Mean and variances after bilateral filtering
#final mean-  231.03541894766698 202.72631565343588 243.82803874643685
#final strd-  10.148999502381884 11.917245295324795 2.738273213838288


# In[13]:


ans_b3=gaussian(x, mb3, sb3)
ans_g3=gaussian(x, mg3, sg3)
ans_r3=gaussian(x, mr3, sr3)
#ans_k=gaussian(x, mk, sk)


# In[14]:


plt.plot(ans_b3, 'b')
print(max(ans_b3))
#plt.show()

plt.plot(ans_g3, 'g')
print(max(ans_g3))
#plt.show()


plt.plot(ans_r3, 'r')
print(max(ans_r3))

plt.show()


# In[16]:


c=cv2.VideoCapture("/home/sanket/Desktop/ENPM673/Untitled Folder/detectbuoy.avi")


while (True):
    ret,image=c.read()
    image_g=image[:,:,1]
    image_r=image[:,:,2]
    if ret == True:
        img_out3=np.zeros(image_g.shape, dtype = np.uint8)
        
        for i in range(0,image_g.shape[0]):
            for j in range(0,image_g.shape[1]):
                z=image_g[i][j]
                if ans_r3[z]>0.06 and ans_g3[z]<0.02 and ans_b3[z]<0.02 and image_r[i][j]<200:
                    img_out3[i][j]=255
                else:
                    img_out3[i][j]=0  
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
                    cv2.circle(image,center,radius,(0,0,255),2)
                    
        cv2.imshow("Threshold",dilation3)
        cv2.imshow('YoYo1', image)
        k = cv2.waitKey(15) & 0xff
        if k == 27:
            break      # wait for ESC key to exit

    else:
        break
        
c.release()
cv2.destroyAllWindows()   


""""cap = cv.VideoCapture('vtest.avi')
fgbg = cv.createBackgroundSubtractorMOG2()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv.imshow('frame',fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()"""


# In[ ]:





# In[ ]:





# In[ ]:




