
"""
This code is to test the DF attack without any defense and it should produce accuracy of arount 98%
"""

# Import necessary modules
import pickle
import numpy as np
from keras.models import load_model
from keras.optimizers import Adamax
from keras.utils import np_utils
from Model_NoDef import DFNet
from tensorflow.python.keras import Sequential

# Define constants
LENGTH = 5000
NB_CLASSES = 95
INPUT_SHAPE = (LENGTH, 1)

# Load test data
with open('D:\\dataset\\ClosedWorld\\NoDef\\X_test_NoDef.pkl', 'rb') as handle:
    X_test1 = np.array(pickle.load(handle, encoding="bytes"))
    
with open('D:\\dataset\\ClosedWorld\\NoDef\\y_test_NoDef.pkl', 'rb') as handle:
    y_test = np.array(pickle.load(handle, encoding="bytes"))

# Preprocessing test data
X_test1 = X_test1[:, :LENGTH]
X_test1 = X_test1[:, :, np.newaxis]

# Load the model architecture
model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=Adamax(), metrics=["accuracy"])

# Load pre-trained weights
model.load_weights('D:\\dataset\\nodef_model_weights_trainer.h5')

# Convert labels to one-hot encoding
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# Evaluate the model on test data
score_test = model.evaluate(X_test1, y_test, verbose=2)
print("Testing accuracy:", score_test[1])
