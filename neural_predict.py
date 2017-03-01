# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:05:19 2017

@author: ae425812
"""
from __future__ import print_function
import numpy as np
from skimage.transform import rotate

import numpy as np
from sklearn.metrics import accuracy_score
from skimage import io
from skimage import data
import  sklearn.preprocessing 
from keras.models import load_model

X_train = np.loadtxt("x_train.csv",delimiter=',',skiprows=1)
y_train = np.loadtxt("y_train.csv",delimiter=',',skiprows=1)
X_test = np.loadtxt("x_test.csv",delimiter=',',skiprows=1)

#we have to skip first column because first column is geneID which is not a feature
X_train =  X_train[:,1:]  #Training data
X_test =  X_test[:,1:]  #Training data
y_train = y_train[:,1:].ravel() #Labels

#Each Gene ID has 100 rows of histone modification in x_train
#Each geneID has label for 100 rows
gene_train = X_train.shape[0]/100  #  total_no_of_rows/100
gene_test = X_test.shape[0]/100

print(gene_train)
X_train = np.array([[x] for x in X_train]) # or list(map([lambda x: [x], X_train]))
X_test = np.array([[x] for x in X_test]) # or list(map([lambda x: [x], X_test]))

X_train = np.split(X_train,gene_train)
X_test = np.split(X_test,gene_test)
print(X_train[0])

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)


y_train = np.vstack([1-y_train, y_train]).T

print(y_train.shape)
print(y_train)

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.core import Dense, Activation
def question():
	N = 10 # Number of feature maps
	w, h = 1, 5 # Conv. window size
	model = Sequential()
	model.add(Convolution2D(nb_filter = N,
	nb_col = w,
	nb_row = h,
	border_mode = 'same',
	activation = 'relu',
	input_shape = (100,1,5)))
	model.add(MaxPooling2D((2,1)))
	
	model.add(Convolution2D(nb_filter = 2*N,
	nb_col = w,
	nb_row = h,
	border_mode = 'same',
	activation = 'relu'))
	model.add(MaxPooling2D((2,1)))
	
	model.add(Convolution2D(nb_filter = 2*N,
	nb_col = w,
	nb_row = h,
	border_mode = 'same',
	activation = 'relu'))
	model.add(MaxPooling2D((2,1)))
	
	
	
	model.add(Flatten())
	model.add(Dense(2, activation = 'sigmoid'))
	
	model.compile(loss='binary_crossentropy', optimizer='sgd') #metrics=['accuracy']
	model.fit(X_train, y_train, nb_epoch=17, batch_size=16) # takes a few seconds
	
	# model.save("neuro_tajarib.h5")
	
	# model = load_model('neuro_tajarib.h5')
	
	y_pred = model.predict_proba(X_test)
	y_pred_classes = model.predict_classes(X_test)
	y_pred_kaggle = np.array(list(map(lambda x: x[1],y_pred)))
	
	# score = accuracy_score(y_test, y_pred_classes)
	# print("the score is",score)
	
	geneId=0
	f = open("kaggle.csv","w")
	f.write("GeneId,prediction")
	f.write("\n")
	for i in y_pred_kaggle:
		geneId = geneId + 1
		f.write(str(geneId)+","+str(i))
		f.write("\n")

	f.close()
question()

