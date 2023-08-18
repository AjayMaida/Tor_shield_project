
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers import ELU
#from keras.layers.advanced_activations import ELU
from keras.initializers import glorot_uniform

class DFNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        model.add(Conv1D(input_shape=input_shape,filters=32,kernel_size=8,padding="same", activation="relu"))
        model.add(Conv1D(filters=32,kernel_size=8,padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2,strides=2))
        model.add(Conv1D(filters=64, kernel_size=8, padding="same", activation="relu"))
        model.add(Conv1D(filters=64, kernel_size=8, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2,strides=2))
        model.add(Conv1D(filters=128, kernel_size=8, padding="same", activation="relu"))
        model.add(Conv1D(filters=128, kernel_size=8, padding="same", activation="relu"))
        model.add(Conv1D(filters=128, kernel_size=8, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2,strides=2))
        model.add(Conv1D(filters=256, kernel_size=8, padding="same", activation="relu"))
        model.add(Conv1D(filters=256, kernel_size=8, padding="same", activation="relu"))
        model.add(Conv1D(filters=256, kernel_size=8, padding="same", activation="relu"))
        model.add(Conv1D(filters=256, kernel_size=8, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2,strides=2))
        model.add(Conv1D(filters=256, kernel_size=8, padding="same", activation="relu"))
        model.add(Conv1D(filters=256, kernel_size=8, padding="same", activation="relu"))
        model.add(Conv1D(filters=256, kernel_size=8, padding="same", activation="relu"))
        model.add(Conv1D(filters=256, kernel_size=8, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2,strides=2))
        model.add(Flatten())
        model.add(Dense(units=1024,activation="relu"))
        model.add(Dense(units=1024,activation="relu"))
        model.add(Dense(classes, activation="softmax"))
        return model