# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:15:38 2020

@author: Suat
"""

#import load_iris function
#iris dataset is already imported in sklearn library
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
from collections import defaultdict

iris = load_iris()

#print (iris.data)
#print (iris.feature_names)
#print (iris.target) #integer forms of iris species
#print (iris.target_names) #0=setosa, 1=versicolor, 2= virginica

X = iris.data #feature matrix
y = iris.target #store response

#let's split train and test data with same sizes as our 
#"manual" example --see the other code in the same folder.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=12)

knn = KNeighborsClassifier(n_neighbors=5) #inctance of k-neighbours classifier

knn.fit(X_train,y_train) #fit the model

#predict the responses and count them
#in the iris.data setosa = 0, versicolor = 1 and virginica = 2, run iris.data
flowers = ["setosa", "versicolor", "virginica"]
num_correct = 0
confusion_matrix: Dict[Tuple, int] = defaultdict(int)


for i in range(len(X_test)):
    prediction = knn.predict([X_test[i]])
    actual = y_test[i]
    
    if prediction == actual:
        num_correct += 1
        
    confusion_matrix[(flowers[prediction[0]], flowers[actual])] += 1

pct_correct = num_correct / len(X_test)
print(pct_correct, confusion_matrix)
