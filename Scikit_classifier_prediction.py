#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 16:01:19 2018

@author: jessicasaini
"""
from sklearn import tree
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#[height,weight, shoesize]
X=[[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,70,40],[159,78,90],[171,75,42],[181,85,43]]
Y=['male','female','female','female','male','male','male','female','male','female']
clf=tree.DecisionTreeClassifier()
clf=clf.fit(X,Y)
prediction=clf.predict([[190,70,43]])
print('Decision Tree',prediction)

#Logistic Regression
logreg=linear_model.LogisticRegression()
logreg.fit(X,Y)
Z=logreg.predict([[190,70,43]])
print('Logistic Regression Prediction',Z)
#Naive Bayes Gaussian NB
gauss=GaussianNB()
gauss.fit(X,Y)
Z=gauss.predict([[190,70,43]])
print('Naive Bayes Prediction:',Z)
#Using SVM
svm=SVC()
svm.fit(X,Y)
Z=svm.predict([[190,70,43]])
print('SVM',Z)
#using KNN
knn=KNeighborsClassifier()
knn.fit(X,Y)
Z=knn.predict([[190,70,43]])
print('KNN',Z)
#using LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis()
lda.fit(X,Y)
Z=lda.predict([[190,70,43]])
print('LDA',Z)


"""
Conclusion: 

    Decision Tree ['female']
Logistic Regression Prediction ['female']
Naive Bayes Prediction: ['male']
SVM ['female']
KNN ['male']
LDA ['male']
    
"""



