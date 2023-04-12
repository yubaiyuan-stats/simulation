#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import keras
import numpy as np
from numpy import linalg as LA
def outcome_loss_fn(y_true,y_pred):
        a1 = tf.reduce_sum(tf.square(y_true[:,0] - y_pred[:,0]))
        a2 = tf.reduce_sum(tf.square(y_true[:,1:] - y_pred[:,1:]))
        return a1 + a2 


def Loss_update_AE(autoencoder,M_Y,treatment,res_Y,res_M,lambda_1,lambda_2,n,m,sess):

    T_M_Y_hat = autoencoder.predict(M_Y)[2]
    bce = tf.keras.losses.BinaryCrossentropy()
    
    loss_T = bce(treatment,T_M_Y_hat[:,0].reshape(n,1)).eval(session=sess)      
    
    Loss_Y_M_T = np.concatenate((treatment - T_M_Y_hat[:,0].reshape(n,1),res_Y - T_M_Y_hat[:,1].reshape(n,1),res_M - T_M_Y_hat[:,2:].reshape(n,m)),axis=1)
    
    corr_res = LA.norm(np.corrcoef(Loss_Y_M_T.transpose()) - np.diag(np.ones(m+2)))**2
    
    total_loss = outcome_loss_fn(M_Y,T_M_Y_hat[:,1:]).eval() + lambda_1*corr_res + lambda_2*loss_T
    return total_loss, T_M_Y_hat

