from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from  sklearn.model_selection import train_test_split
import csv
import numpy as np
import os

data_path = "Data"


x_train = np.loadtxt(data_path + os.sep + "x_train.csv",delimiter=',',skiprows=1)
y_train = np.loadtxt(data_path + os.sep + "y_train.csv",delimiter=',',skiprows=1)
x_test = np.loadtxt(data_path + os.sep + "x_test.csv",delimiter=',',skiprows=1)
#we have to skip first column because first column is geneID which is not a feature
x_train =  x_train[:,1:]  #Training data
y_train = y_train[:,1:]  #Labels
x_test =  x_test[:,1:] #Test data

#Each Gene ID has 100 rows of histone modification in x_train
#Each geneID has label for 100 rows

gene_train = x_train.shape[0]/100  #  total_no_of_rows/100
gene_test = x_test.shape[0]/100



x_train = np.split(x_train,gene_train)
x_test = np.split(x_test,gene_test)

#x_train , x_test is a multidimensional vector
#flatten it into 1D vector

x_train = [i.ravel() for i in x_train]
x_test= [j.ravel() for j in x_test]

#convert it into array

x_train = np.array(x_train)
x_test = np.array(x_test)

logr = linear_model.LogisticRegression()

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.4, random_state=0)
logr.fit(x_train,y_train)
y_pred = logr.predict(x_test)
print (roc_auc_score(y_test,y_pred))

print("ali")



