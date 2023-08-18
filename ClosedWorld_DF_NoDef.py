#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code is to implement deep fingerprinting model for website fingerprinting attacks
# ACM Reference Formant
# Payap Sirinam, Mohsen Imani, Marc Juarez, and Matthew Wright. 2018.
# Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning.
# In 2018 ACM SIGSAC Conference on Computer and Communications Security (CCS â€™18),
# October 15â€“19, 2018, Toronto, ON, Canada. ACM, New York, NY, USA, 16 pages.
# https://doi.org/10.1145/3243734.3243768


from keras import backend as K
from utility import LoadDataNoDefCW
from Model_NoDef import DFNet
import random
from keras.utils import np_utils
from keras.optimizers import Adamax
import numpy as np
import os
import pickle

random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Use only CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

description = "Training and evaluating DF model for closed-world scenario on non-defended dataset"

print(description)
# Training the DF model
NB_EPOCH = 30   # Number of training epoch
print ("Number of Epoch: ", NB_EPOCH)
BATCH_SIZE = 128 # Batch size
VERBOSE = 2 # Output display mode
LENGTH = 5000 # Packet sequence length
OPTIMIZER = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer

NB_CLASSES = 95 # number of outputs = number of classes
INPUT_SHAPE = (LENGTH,1)


# Data: shuffled and split between train and test sets
print ("Loading and preparing data for training, and evaluating the model")
X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataNoDefCW()
# Please refer to the dataset format in readme
#TODO :
#K.tensorflow_backend.set_image_dim_ordering("tf") # tf is tensorflow
#K.common.set_image_dim_ordering('tf')
K.set_image_data_format('channels_last')
print(X_train[1].tolist())
for i in range(len(X_train)):
    for j in range(len(X_train[i])):
        if X_train[i][j]==1:
            l=X_train[i][j:].tolist()
            for k in range(len(l),5000):
                l.append(0)
            X_train[i]=np.array(l[:])
            break
for i in range(len(X_test)):
    for j in range(len(X_test[i])):
        if X_test[i][j]==1:
            l=X_test[i][j:].tolist()
            for k in range(len(l),5000):
                l.append(0)
            X_test[i]=np.array(l[:])
            break
for i in range(len(X_valid)):
    for j in range(len(X_valid[i])):
        if X_valid[i][j]==1:
            l=X_valid[i][j:].tolist()
            for k in range(len(l),5000):
                l.append(0)
            X_valid[i]=np.array(l[:])
            break
# Convert data as float32 type
print(X_train[1].tolist())

'''
# ajay uncommented this code (from line 78 to line 107)
filenew2=open('D:\\dataset\\models\\trainery_nodef.pkl','wb')
filenew3=open('D:\\dataset\\models\\attackerx_nodef.pkl','wb')
filenew4=open('D:\\dataset\\models\\attackery_nodef.pkl','wb')
filenew1=open('D:\\dataset\\models\\trainerx_nodef.pkl','wb')
datax=[]
datay=[]
datadx=[]
datady=[]
count_array=[0]*95
for i in range(len(X_train)):
    if count_array[y_train[i]]<400:
        datax.append(X_train[i])
        datay.append(y_train[i])
    else:
        datadx.append(X_train[i])
        datady.append(y_train[i])
    count_array[y_train[i]]+=1
datax=np.array(datax)
datay=np.array(datay)
datadx=np.array(datadx)
datady=np.array(datady)
pickle.dump(datax,filenew1)
pickle.dump(datay,filenew2)
pickle.dump(datadx,filenew3)
pickle.dump(datady,filenew4)
print(datax.shape)
print(datay.shape)
print(datadx.shape)
print(datady.shape)
print('dump complete')'''

X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
y_test = y_test.astype('float32')

# we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
X_train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]
X_test = X_test[:, :,np.newaxis]

print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'validation samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to categorical classes matrices
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_valid = np_utils.to_categorical(y_valid, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# Building and training model
print ("Building and training DF model")

model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
print(model.summary())
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
	metrics=["accuracy"])
print ("Model compiled")

# Start training
history = model.fit(X_train, y_train,
		batch_size=BATCH_SIZE, epochs=NB_EPOCH,
		verbose=VERBOSE, validation_data=(X_valid, y_valid),shuffle=True)

model.save('D:\\dataset\\nodef_model_upgraded.h5')
model.save_weights('D:\\dataset\\nodef_model_weights_upgraded.h5')
# Start evaluating model with testing data
score_test = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("Testing accuracy:", score_test[1])


