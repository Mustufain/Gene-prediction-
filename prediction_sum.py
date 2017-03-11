from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from  sklearn.cross_validation import train_test_split
import csv
import numpy as np
import os

data_path = ""

x_train = np.loadtxt("x_train.csv",delimiter=',',skiprows=1)
y_train = np.loadtxt("y_train.csv",delimiter=',',skiprows=1)
#we have to skip first column because first column is geneID which is not a feature
x_train =  x_train[:,1:]  #Training data
y_train = y_train[:,1:].ravel() #Labels

#Each Gene ID has 100 rows of histone modification in x_train
#Each geneID has label for 100 rows
gene_train = x_train.shape[0]/100  #  total_no_of_rows/100


x_train = np.split(x_train,gene_train)
x2_train = [[sum(row[j] for row in gene) for j in range(5)] for gene in x_train]


x2_train = np.array(x2_train)

print("x_train shape is %s" % str(x2_train.shape))    
print("y_train shape is %s" % str(y_train.shape))
logr = linear_model.LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(x2_train, y_train, test_size=0.4, random_state=0)
logr.fit(x_train,y_train)
y_pred = logr.predict(x_test)
print(roc_auc_score(y_test,y_pred))
