import csv
import os
f = open('train.csv','w',encoding='utf-8',newline="")
csv_write = csv.writer(f,dialect='excel')
filePath = 'F:\\BaiduNetdiskDownload\\dataset\\train\\'
tmp = 1
for i,j,k in os.walk(filePath):

    stu1 = i+"/"
    stu2 = i+"/"
    stu1 =stu1+"".join(k[0:1])
    stu2 =stu2+"".join(k[1:2])

    if tmp == 1:
        tmp += 1
        continue
    else:
        csv_write.writerow([stu1,stu2])

