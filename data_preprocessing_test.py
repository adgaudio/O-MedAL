import os
import PIL
from PIL import Image
import numpy as np
fin = []
finl = []
path = "/home/kartik/Downloads/18980/val/val/normal"
files = os.listdir(path)
for file in files:
   a = Image.open(path+"/"+file).convert("RGB")
   label = 0
   a = a.resize((512,512),Image.ANTIALIAS)
   fin.append(np.asarray(a))
   finl.append(label)
path = "/home/kartik/Downloads/18980/val/val/pathological/"
files = os.listdir(path)
for file in files:
   a = Image.open(path+"/"+file).convert("RGB")
   label = 1
   a = a.resize((512,512),Image.ANTIALIAS)
   fin.append(np.asarray(a))
   finl.append(label)
np.save("X_Test.npy",fin)
np.save("Y_Test.npy",finl)
