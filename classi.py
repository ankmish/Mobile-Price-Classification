# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 10:34:51 2018

@author: AkM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

dataset=pd.read_csv('train.csv')
df = pd.read_csv('test.csv')
df.drop('id',axis =1 ,inplace = True)
X=dataset.drop('price_range',axis=1)
y=dataset['price_range']
X_test = df.iloc[:,:].values

"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

"""
from sklearn.neighbors import KNeighborsClassifier 
parameters = {'n_neighbors':[1,20]}
knn = KNeighborsClassifier(p=2,n_jobs=20)
clf=GridSearchCV(knn,parameters)
clf.fit(X,y)
y_pred = clf.predict(X_test)
"""

from sklearn import datasets, svm, tree
from sklearn.model_selection import train_test_split, GridSearchCV
svc = svm.SVC()
parameters = {'kernel':('linear','rbf'),'C':[1,40]}
clf=GridSearchCV(svc,parameters)
model = clf.fit(X_train,y_train)
y_pred = model.predict(X_test)


"""
#accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
"""

np.savetxt('y_pred_11.csv',y_pred ,delimiter=',')
