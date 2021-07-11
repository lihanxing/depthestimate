
import os
import time

filePath = './dataset/val/'
tmp = 0
a = 1
for i,j,k in os.walk(filePath):
    if a==1:
        a+=1
        continue

    if a>=4:
        break
    path = r"./result/"
    path = path+i[14:]
    os.makedirs(path)
    os.system('python test.py '+"--model "+"models/1625803710-n875-e20-bs8-lr0.0001-densedepth_nyu/weights.20-69.85.hdf5 "+"--input "+i+"/color.jpg "+"--output "+path)
    # tmp+=1
   
   

