import os
import json
import pickle
import numpy as np
from hpelm import ELM
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

NO_AP_MAX = 7

class Regressor(object):
    def __init__(self):
        pass

    def support_vector_regressor(self, kernel, degree, gamma, coef0, tol, C, epsilon):
        return SVR(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C, epsilon=epsilon)

    def knn_regressor(self, n_neighbors):
        return KNeighborsRegressor(n_neighbors=n_neighbors)

    def elm_regressor(self, ap, neurons=[], activation_function=[]):
        elm = ELM(ap, 2, classification='r')
        for i in range(len(neurons)):
            elm.add_neurons(neurons[i], activation_function[i])
        return elm

    def dnn_regressor(self, ap, units, hiddens, activation, dropout):
        input_layer = Input(shape=(ap,), name='input')
        temp_layer = input_layer
        for _ in range(hiddens):
            temp_layer = Dense(units=units, activation=activation)(temp_layer)
            temp_layer = Dropout(dropout)(temp_layer)
            temp_layer = BatchNormalization()(temp_layer)
            output = Dense(units=2, activation=activation, name='output')(temp_layer)
            model = Model(input_layer, output)
        print (model.summary())
        return model

    def crnn_regressor(self):
        pass

if __name__ == '__main__':
    regressor = Regressor()
    dnn = regressor.dnn_regressor(ap=7, units=50, hiddens=5, activation='elu', dropout=0.1)