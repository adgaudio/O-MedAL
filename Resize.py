from __future__ import print_function
import numpy as np 
from PIL import Image as I 
import cv2


Data = np.load('Orig/Main_Data.npy',encoding='bytes')
Test = np.load('Orig/X_Test.npy',encoding='bytes')

Resized_data = np.zeros([1,3,200,200])
for e,i in enumerate(Data):
    print ('Data:{}'.format(e))
    i = np.reshape(i,(512,512,3))
    Resized_data = np.concatenate((Resized_data,np.reshape(cv2.resize(i,(200,200),0.0, 0.0, interpolation=cv2.INTER_CUBIC),(1,3,200,200))),axis=0)

Resized_data = Resized_data[1:,:,:,:]

np.save('Main_Data1.npy',Resized_data)    

Resized_data = np.zeros([1,3,200,200])

for e,i in enumerate(Test):
    print ('Test:{}'.format(e))
    i = np.reshape(i,(512,512,3))
    Resized_data = np.concatenate((Resized_data,np.reshape(cv2.resize(i,(200,200),0.0, 0.0, interpolation=cv2.INTER_CUBIC),(1,3,200,200))),axis=0)

Resized_data = Resized_data[1:,:,:,:]

np.save('X_Test1.npy',Resized_data) 