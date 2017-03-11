from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from  sklearn.model_selection import train_test_split
import csv
import numpy as np
import os
from sklearn.model_selection import cross_val_score

data_path = "Data"
#15485 genes

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
penalty = ["l1","l2"]
ori_test = x_test
logr = linear_model.LogisticRegression()
predicted_class=[]
#x1_train, x1_test, y1_train, y1_test = train_test_split(x_train, y_train, test_size=0.1, random_state=0)
#We have 2 classes 0 and 1
#C_range = 10.0 ** np.arange(-4, 3)
#for c in C_range:
c= 0.10000000000000001
logr.penalty = penalty[0]
logr.C = c
logr.fit(x_train,y_train)
#print (roc_auc_score(y1_test,logr.predict(x1_test)))
y_pred = logr.predict_proba(x_test)
#print (roc_auc_score(y1_test,y_pred))

# penalty = l1 c =  0.10000000000000001  ---> 0.85395
# penalty = l2 c =    ---->

geneId=0
#print(y_pred.shape)
f = open("kaggle.csv","r+")
f.write("GeneId,prediction")
f.write("\n")
for i in y_pred:

    geneId = geneId + 1
    f.write(str(geneId)+","+str(i[1]))
    f.write("\n")



f.close()





