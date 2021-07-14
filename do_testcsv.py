from numpy import testing


#生成test.csv
import csv
import os
f = open('test.csv','w',encoding='utf-8',newline="")
csv_write = csv.writer(f,dialect='excel')
filePath = './dataset/val/'
tmp = 1
for i,j,k in os.walk(filePath):

    stu1 = i+"/"
    stu2 = i+"/"
    stu1 =stu1+"".join(k[1:2])
    stu2 =stu2+"".join(k[3:4])
    stu1 = "dataset/val/"+stu1
    stu2 = "dataset/val/"+stu2

    if tmp == 1:
        tmp += 1
        continue
    else:
        csv_write.writerow([stu1,stu2])

