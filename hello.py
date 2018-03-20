import sys
# Takes first name and last name via command 
# line arguments and then display them
#print("Output from Python")
#print(sys.argv[1])
#print(sys.argv[2])
#print(sys.argv[3])
#print(sys.argv[4])
#print(sys.argv[5])
#print(sys.argv[6])
#print(sys.argv[7])
#print(sys.argv[8])
#print(sys.argv[9])
#print(sys.argv[10])
#print("hello")
# save the script as hello.py
month=sys.argv[11]
place=sys.argv[12]

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt


placeurl='allcsv/'+place+'/';
avgtmp= pd.read_csv(placeurl+'avg.csv',sep='\t')
cloud=pd.read_csv(placeurl+'cc.csv',sep='\t')
dr=pd.read_csv(placeurl+'dr.csv',sep='\t')
grf=pd.read_csv(placeurl+'gff.csv',sep='\t')
maxtemp=pd.read_csv(placeurl+'max.csv',sep='\t')
mintemp=pd.read_csv(placeurl+'min.csv',sep='\t')
pevap=pd.read_csv(placeurl+'pevap.csv',sep='\t')
wdf=pd.read_csv(placeurl+'wdf.csv',sep='\t')
rce=pd.read_csv(placeurl+'rce.csv',sep='\t')
vp=pd.read_csv(placeurl+'vp.csv',sep='\t')


rainfall=pd.read_csv(placeurl+'rainfall.csv',sep='\t')


res=[avgtmp[month],cloud[month],dr[month],grf[month],maxtemp[month],mintemp[month],pevap[month],wdf[month],rce[month],vp[month],rainfall[month]]

jan = pd.concat(res,axis=1)
jan.columns=['avgtemp','cloud','dr','grf','maxtemp','mintemp','pevap','wdf','rce','vp','rainfall']

from sklearn.model_selection import train_test_split

X=pd.DataFrame(data=jan.drop(['rainfall'],axis=1),columns=['avgtemp','cloud','dr','grf','maxtemp','mintemp','pevap','wdf','rce','vp'])

Y=pd.DataFrame(data=jan['rainfall'],columns=['rainfall'])

#X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
X_test=X[0:1]
y_test=Y[0:1]
#print(X_test)
X_train=X
y_train=Y

test_list= {'avgtemp':[sys.argv[1]],'cloud':[sys.argv[2]],'dr':[sys.argv[3]],'grf':[sys.argv[4]],'maxtemp':[sys.argv[5]],'mintemp':[sys.argv[6]],'pevap':[sys.argv[7]],'wdf':[sys.argv[8]],'rce':[sys.argv[9]],'vp':[sys.argv[10]]}
X_test=pd.DataFrame(data=test_list,columns=['avgtemp','cloud','dr','grf','maxtemp','mintemp','pevap','wdf','rce','vp'])
#print(test_list)
#print(X_test)
#print(X_train[0:1])
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

from sklearn.svm import SVR

SVM=SVR(kernel='rbf',C=1e3, gamma=0.1)

SVM.fit(X_train,y_train)

y_pred=SVM.predict(X_test)

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))

#print(X_test)
print(y_pred)