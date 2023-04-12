#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def Out_sample_pred(treatment,M,X,rf_label,test_X,test_M,test_treatment,beta_Y,beta_Y_M,beta_Y_T,gamma_Y_bias,test_Y,predict_track,n,n1,method):

##use random forest to estimate U for test data
    # rf_train = np.concatenate((treatment,Loss_M_pred),axis=1)
    # rf_label = Y_bias
    # rf = RandomForestRegressor(n_estimators = 100, max_depth=12,random_state = 123)
    # rf.fit(rf_train,rf_label.reshape(n,))
    
    # Loss_M_U_test = test_M - (np.matmul(test_X[:,0:5],beta_M.transpose()).reshape(n1,2) +  test_treatment*T_M)
    # Y_bias_pred = rf.predict(np.concatenate((test_treatment,Loss_M_U_test),axis=1)).reshape(n1,1)
    # pred_Y = np.matmul(test_X[:,5:],beta_Y) + np.matmul(test_M,beta_Y_M) + test_treatment*beta_Y_T + gamma_Y_bias*Y_bias_pred.reshape(n1,1)
    # pred = (sum((pred_Y.reshape(n1,1) - test_Y.reshape(n1,1))**2)/n1)**0.5 
    # return pred


    rf_train = np.concatenate((treatment,M,X),axis=1)
    if method == 'low_rank':
       rf = RandomForestRegressor(n_estimators = 50, max_depth=12,random_state = 123)
    else:
       rf = RandomForestRegressor(n_estimators = 200, max_depth=15,random_state = 123)
    rf.fit(rf_train,rf_label.reshape(n,))
    
    Y_bias_pred = rf.predict(np.concatenate((test_treatment,test_M,test_X),axis=1)).reshape(n1,1)
    pred_Y = np.matmul(test_X[:,5:],beta_Y) + np.matmul(test_M,beta_Y_M) + test_treatment*beta_Y_T + gamma_Y_bias*Y_bias_pred.reshape(n1,1)
    pred_1 = (sum((pred_Y.reshape(n1,1) - test_Y.reshape(n1,1))**2)/n1)**0.5 
    predict_track.append((sum((pred_Y.reshape(n1,1) - test_Y.reshape(n1,1))**2)/n1)**0.5)
    pred = min(predict_track)
    if method == 'low_rank':
       return pred
    else:
       return pred_1
    

 