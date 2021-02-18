# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:30:39 2021

@author: TokTam
"""

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from IPython.display import Image
import pydotplus
from sklearn import tree
import collections

File = pd.read_csv("C:\\New folder\\Diabetes_Diagnosis.csv")
print(File)
'''
for creating a matrix
'''
def fcam(diabetes):
    if diabetes == True:
        diabetes = '1'
    if diabetes == False:
        diabetes = '0'
    return diabetes

Z = File.diabetes.apply(fcam)
print(Z)

'for spliting data; first Select our featuresas a list and than  define our culmn and rows'
features = ['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','age','diab_pred','skin']
X = File[features]    #As culomns
Y = File.diabetes.apply(fcam)     #As rows
''' Now we can split our data with train_test_split'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=1)
'''
Now we can build our model using classifiar
'''
Class = DecisionTreeClassifier()
CLSFR = Class.fit(X_train , Y_train)
'''
Our prediction
''' 
Y_pred = CLSFR.predict(X_test)
print(Y_pred)
'''
Using sklearn.metrics for confusion matrix
'''
confusion_matrix(Y_test, Y_pred)
print(confusion_matrix(Y_test, Y_pred))
# Define Accuracy
def accuracy(Y_test , Y_pred):
    Corrects = 0
    for i in range(len(Y_pred)):
         if int( Y_test[i]) is int(Y_pred[i]):
             Corrects += 1
             Acc = float( Corrects / len(Y_test))*100
             print(Acc)
#Also we can use the below code as Follow
Accuracy = metrics.accuracy_score(Y_test , Y_pred)*100
print(Accuracy)
# importing Data to decision tree and create it.....
#Decision Tree Data
from sklearn.tree import export_graphviz
dot_data = tree.export_graphviz(CLSFR, out_file=None,  
                filled=True, rounded=True,
                special_characters=True, feature_names = features,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data)  
colors = ('orange', 'yellow')
edges = collections.defaultdict(list)
for edge in graph.get_edge_list():
    edges[edge.get_source() ].append(int(edge.get_destination()))
for edge in edges: edges[edge].sort()
for i in range(2):dest = graph.get_node(str(edges[edge][i]))[0]
dest.set_fillcolor(colors[i])
graph.write_png('diabetes.png')
from pandas import cut
print(pd.cut(File.bmi , bins=3 , right=False))







y = '((toktam pazesh))'
z = y.center(67,'0')
print(z)