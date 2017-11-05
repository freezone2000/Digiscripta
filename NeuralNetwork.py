# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 05:40:28 2017

@author: MLH Admin
"""
from __future__ import print_function


import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Conv2D, MaxPooling1D, AveragePooling1D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras import backend as K



#first get the data (repeat process for each letter)
dataframe = pd.read_csv('testA.csv')

nc = None
  
epochs = 12
  

#input dimensions
n_cols = 50
n_rows = 120*70


train_a = dataframe.iloc[:100,1:351]
test_a = dataframe.iloc[100:120,1:351]
valid_a = dataframe.iloc[:100,0]
valid_a2 = dataframe.iloc[100:120,0]
test_y = dataframe.iloc[0:120,0:1]

#train_a = keras.utils.to_categorical(train_a, num_classes = nc)
#print(train_a)
#test_a  = keras.utils.to_categorical(test_a, num_classes = nc)
'''  
dataframe = pd.read_csv('testB.csv')
train_b = dataframe[:100,1:351]
test_b = dataframe[100:120,1:351]
valid_b = dataframe[:100,0]
valid_b2 = dataframe[100:120,0]


dataframe = pd.read_csv('testC.csv')
train_c = dataframe[:100,1:351]
test_c = dataframe[100:120,1:351]
valid_c = dataframe[:100,0]
valid_c2 = dataframe[100:120,0]


dataframe = pd.read_csv('testD.csv')
train_d = dataframe[:100,1:351]
test_d = dataframe[100:120,1:351]
valid_d = dataframe[:100,0]
valid_d2 = dataframe[100:120,0]


dataframe = pd.read_csv('testE.csv')
train_e = dataframe[:100,1:351]
test_e = dataframe[100:120,1:351]
valid_e = dataframe[:100,0]
valid_e2 = dataframe[100:120,0]


dataframe = pd.read_csv('testF.csv')
train_f = dataframe[:100,1:351]
test_f = dataframe[100:120,1:351]
valid_f = dataframe[:100,0]
valid_f2 = dataframe[100:120,0]
'''
  
'''
  train_g = dataframe[:100,1:dataframe.len_column()]
  test_g = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_h = dataframe[:100,1:dataframe.len_column()]
  test_h = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_i = dataframe[:100,1:dataframe.len_column()]
  test_i = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_j = dataframe[:100,1:dataframe.len_column()]
  test_j = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_k = dataframe[:100,1:dataframe.len_column()]
  test_k = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_l = dataframe[:100,1:dataframe.len_column()]
  test_l = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_m = dataframe[:100,1:dataframe.len_column()]
  test_m = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_n = dataframe[:100,1:dataframe.len_column()]
  test_n = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_o = dataframe[:100,1:dataframe.len_column()]
  test_o = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_p = dataframe[:100,1:dataframe.len_column()]
  test_p = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_q = dataframe[:100,1:dataframe.len_column()]
  test_q = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_r = dataframe[:100,1:dataframe.len_column()]
  test_r = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_s = dataframe[:100,1:dataframe.len_column()]
  test_s = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_t = dataframe[:100,1:dataframe.len_column()]
  test_t = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_u = dataframe[:100,1:dataframe.len_column()]
  test_u = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_v = dataframe[:100,1:dataframe.len_column()]
  test_v = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_w = dataframe[:100,1:dataframe.len_column()]
  test_w = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_x = dataframe[:100,1:dataframe.len_column()]
  test_x = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_y = dataframe[:100,1:dataframe.len_column()]
  test_y = dataframe[100:120,1:dataframe.len_column()]
  
  
  train_z = dataframe[:100,1:dataframe.len_column()]
  test_z = dataframe[100:120,1:dataframe.len_column()]
'''
  
  
  
#convert class vectors to binary class matrices

'''
train_b = keras.utils.to_categorical(train_b,num_classes)
test_b  = keras.utils.to_categorical(test_b, num_classes)


train_c = keras.utils.to_categorical(train_c,num_classes)
test_c  = keras.utils.to_categorical(test_c, num_classes)


train_d = keras.utils.to_categorical(train_d,num_classes)
test_d  = keras.utils.to_categorical(test_d, num_classes)


train_e = keras.utils.to_categorical(train_e,num_classes)
test_e  = keras.utils.to_categorical(test_e, num_classes)


train_f = keras.utils.to_categorical(train_f,num_classes)
test_f  = keras.utils.to_categorical(test_f, num_classes)
'''
  

#build model (do PCA through Keras)
model = Sequential()

model.add(MaxPooling1D(pool_size=(2), strides=None, padding='valid',input_shape = (100,351)))

model.add(Dropout(0.20,noise_shape=None, seed=None))

model.add(Flatten())

model.add(Dense(55, activation='relu'))

model.add(Dropout(0.4, noise_shape=None, seed=None))

model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])


#train that shit
model.fit(train_a, test_y,
          batch_size= 32,
          epochs = 10,
          verbose = 1,
          callbacks = None,
          validation_split=0.0,
          validation_data = None,
          shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)

score = model.evaluate(a_test, valid_a, verbose=1, sample_weight=None)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#save the model
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.writer(model_json)
    
model.save_weights("model.h5")