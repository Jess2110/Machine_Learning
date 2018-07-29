
"""
Created on Sun Jul 29 15:58:12 2018

@author: jessicasaini
#Gender Classification
"""


from sklearn import tree
#[height,weight, shoesize]
X=[[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,70,40],[159,78,90],[171,75,42],[181,85,43]]
Y=['male','female','female','female','male','male','male','female','male','female']
clf=tree.DecisionTreeClassifier()
clf=clf.fit(X,Y)
prediction=clf.predict([[190,70,43]])
print(prediction)


