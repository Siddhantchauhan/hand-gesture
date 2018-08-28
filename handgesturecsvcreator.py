import pandas as pd
import numpy as np

import cv2
PIXEL=[]
for i in range(5184):
    PIXEL.append("PIXEL"+str(i))
PIXEL.append('label')
data=[]
def main():
    countfist = 1
    while(countfist<387):

        imgpath = "E:\\data\\five\\"+ str(countfist)+".jpg"
        img = cv2.imread(imgpath)
        imga,g,r=cv2.split(img)

        imga=imga.ravel()
        imga=np.where(imga>0,1,0)
        imga=list(imga)
        imga.append(1)
        data.append(imga)
        countfist+=1
    countfive = 1
    while (countfive < 221):
        imgpath = "E:\\data\\fist\\" + str(countfive) + ".jpg"
        img = cv2.imread(imgpath)
        imga, g, r = cv2.split(img)

        imga = imga.ravel()
        imga = np.where(imga > 0, 1, 0)
        imga = list(imga)
        imga.append(2)
        data.append(imga)
        countfive += 1
    countno=1
    while (countno < 193):
        imgpath = "E:\\data\\no\\" + str(countno) + ".jpg"
        img = cv2.imread(imgpath)
        imga, g, r = cv2.split(img)

        imga = imga.ravel()
        imga = np.where(imga > 0, 1, 0)
        imga = list(imga)
        imga.append(3)
        data.append(imga)
        countno += 1
if __name__ == "__main__":
    main()
df=pd.DataFrame(data,columns=PIXEL)
print(df)
df.to_csv("handgesture1.csv",index=False)