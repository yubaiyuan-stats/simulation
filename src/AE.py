#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import keras
from keras import layers
from src.utli import *
import tensorflow as tf
import tensorflow_probability as tfp




def AE(treatment,X,M,Y,lambda_1,lambda_2,n,m):

    n = n
    Data = np.concatenate((treatment,Y,M),axis=1)
    
    if m == 2:
    
        input_U_1 = keras.Input(shape=(3,))
        
        x1 = layers.Dense(3, activation='selu')(input_U_1)
        x1 = layers.Dense(6, activation='selu')(x1)
        x3 = layers.Dense(10, activation='selu', activity_regularizer = UncorrelatedFeaturesConstraint_target(10,treatment.reshape(n,), weightage = 10))(x1)
        
        
        x2 = layers.Dense(4, activation='selu')(x1)
        x2 = layers.Dense(4, activation='selu')(x2)
        treatment_pred = layers.Dense(1, activation='sigmoid')(x2)
        
        encoded = layers.concatenate([x2,x3])
        
        y = layers.Dense(6, activation='selu')(x3)
        y = layers.Dense(3, activation='selu')(y)
        decoded = layers.Dense(3)(y)
        
    else:    
        
        input_U_1 = keras.Input(shape=(6,))

        x1 = layers.Dense(6, activation='selu')(input_U_1)
        x1 = layers.Dense(12, activation='selu')(x1)
        x3 = layers.Dense(20, activation='selu', activity_regularizer = UncorrelatedFeaturesConstraint_target(20,treatment.reshape(n,), weightage = 20))(x1)
         

        x2 = layers.Dense(4, activation='selu')(x1)
        x2 = layers.Dense(4, activation='selu')(x2)
        treatment_pred = layers.Dense(1, activation='sigmoid')(x2)

        encoded = layers.concatenate([x2,x3])

        y = layers.Dense(12, activation='selu')(x3)
        y = layers.Dense(6, activation='selu')(y)
        decoded = layers.Dense(6)(y)
        
    T_M_Y_hat = layers.concatenate([treatment_pred,decoded])
    
    
    autoencoder = keras.Model(inputs = [input_U_1], outputs=[treatment_pred,decoded,T_M_Y_hat])
    encoder = keras.Model(inputs = [input_U_1], outputs= encoded)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02),loss=["binary_crossentropy",outcome_loss_fn,corr_loss_fn(Data = Data),],loss_weights=[lambda_2,1,lambda_1])

    return autoencoder, encoder




