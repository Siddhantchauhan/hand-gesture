import pickle
import numpy as np
from sklearn import svm
import pandas as pd
from sklearn import model_selection as cv
from sklearn import preprocessing as pp
#from matplotlib import pyplot as plt
PIXEL=[]
for i in range(5184):
    PIXEL.append("PIXEL"+str(i))
df = pd.read_csv('handgesture1.csv')
x = df[PIXEL]
y = df[["label"]]

x = np.array(x)
y = np.array(y)

x=x.reshape(len(x),5184)
y=y.ravel()
#print(x)
#print(y)

model = svm.SVC()

xtrain,xtest,ytrain,ytest= cv.train_test_split(x,y,test_size=0.3)
model.fit(xtrain,ytrain)
accuracy= model.score(xtest,ytest)
print(accuracy)
with open('handgesturesvmmodel.pickle','wb') as f:
     pickle.dump(model,f)