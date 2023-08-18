#Import Modules
import pickle
import numpy as np
import random
from keras.models import load_model
from keras.optimizers import Adamax
from keras.utils import np_utils
from Model_NoDef import DFNet
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
import math
import sys
from scipy.optimize import differential_evolution

# Training the DF model
NB_EPOCH = 10   # Number of training epoch
BATCH_SIZE = 128 # Batch size
VERBOSE = 2 # Output display mode
LENGTH = 5000 # Packet sequence length
OPTIMIZER = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer

NB_CLASSES = 95 # number of outputs = number of classes
INPUT_SHAPE = (LENGTH,1)
'''def predict_func(x):
    global dump_x
    global index
    global clss
    global d_x
    global index1
    dump_x1=dump_x[:]
    index1=0
    for j in range(index1,index1+index):
        if j>=5000:
            break
        if x[j-index1]<=0.5 and x[j-index1]>=-0.5:
            dump_x1[j]=0.0
        elif x[j-index1]>0.5:
            dump_x1[j]=1.0
        else:
            dump_x1[j]=-1.0
        
    dump_x2=dump_x1[:]
    dump_x1=np.array([dump_x1])
    dump_x1=dump_x1.astype('float32')
    dump_x1=dump_x1[:,:,np.newaxis]
    prediction=model.predict(dump_x1)[:,0]
    q=model.predict_classes(dump_x1)
    if q!=clss:
        print(q,end=' ')
        for l in range(0,5000-index):
            dump_x2[l+index]=d_x[l]
        dump_x3=dump_x2
        dump_x2=np.array([dump_x2])
        dump_x2=dump_x2.astype('float32')
        dump_x2=dump_x2[:,:,np.newaxis]
        prediction=model.predict(dump_x2)[:,0]
        qm=model.predict_classes(dump_x2) 
        print(qm)
        print(dump_x3)
        if 
        sys.exit()
    return prediction'''
def create_model():
    model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,metrics=["accuracy"])
    return model

OPTIMIZER = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer

#Opening the Traces generated
with open('D:\\dataset\\GAT\\Adversarial_Traces_x_alpha30_only1.pkl', 'rb') as handle:
	X_test = np.array(pickle.load(handle,encoding="bytes"))

#Testing on DeepFingerprinting Attacker Model
with open('D:\\dataset\\ClosedWorld\\NoDef\\X_test_NoDef.pkl', 'rb') as handle:
	X_test1 = np.array(pickle.load(handle,encoding="bytes"))
with open('D:\\dataset\\ClosedWorld\\NoDef\\y_test_NoDef.pkl', 'rb') as handle:
	y_test = np.array(pickle.load(handle,encoding="bytes"))
print(y_test.shape)
count=0
'''for j in range(len(X_test)):
    for i in range(len(X_test[j])):
        if X_test[j][i]==0:
            count+=i
            break
print(count)'''
#print(len(X_test_old[0]))

model=create_model()
model.load_weights('D:\\dataset\\nodef_model_weights_trainer.h5')
X_test1=[]
count=0

for j in range(len(X_test)):
    x=[]
    for i in X_test[j]:
        x.append(i)
    x=x[:5000]
    X_test1.append(np.array(x))
X_test1=np.array(X_test1)
print(X_test1.shape,"")
X_test1=X_test1[:,:,np.newaxis]
y_test = np_utils.to_categorical(y_test, NB_CLASSES)
score_test = model.evaluate(X_test1, y_test, verbose=VERBOSE)
print("Testing accuracy:", score_test[1])
'''

for i in range(1,150):
    index=i
    bounds=[(-1,1)]*index
    print(i)
    diff=differential_evolution(predict_func,bounds,popsize=1)
#print(diff.x.tolist())
    dump_x=np.array([dump_x])
    dump_x=dump_x.astype('float32')
    dump_x=dump_x[:,:,np.newaxis]
    print("")
    dump_x=X_test[30].tolist()[:]'''
