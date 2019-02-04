import os
import PIL
from PIL import Image
import numpy as np
fin = []
finl = []
path = "/home/kartik/Downloads/18980/train/normal"
files = os.listdir(path)
for file in files:
   a = Image.open(path+"/"+file).convert("RGB")
   label = 0
   a = a.resize((512,512),Image.ANTIALIAS)
   fin.append(np.asarray(a))
   finl.append(label)
path = "/home/kartik/Downloads/18980/train/pathological"
files = os.listdir(path)
for file in files:
   a = Image.open(path+"/"+file).convert("RGB")
   label = 1
   a = a.resize((512,512),Image.ANTIALIAS)
   fin.append(np.asarray(a))
   finl.append(label)
np.save("Main_Data.npy",fin)
np.save("Main_Labels.npy",finl)
