#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from econml.sklearn_extensions.model_selection import GridSearchCVList

def first_stage_reg():
    return GridSearchCVList([Lasso(),
                              RandomForestRegressor(n_estimators=30, random_state=123),
                              GradientBoostingRegressor(random_state=123)],
                              param_grid_list=[{'alpha': [.001, .01, .1, 1, 10]},
                                                {'max_depth': [3, None],
                                                'min_samples_leaf': [10, 50]},
                                              {'n_estimators': [50, 100],
                                                'max_depth': [3,10],
                                                'min_samples_leaf': [10, 30]}],
                              cv=5,
                              scoring='neg_mean_squared_error')

def Out_sample_pred_1(treatment,M,X,Y_bias,test_X,test_M,beta_M,test_treatment,T_M,beta_Y_ini,beta_Y_M_ini,beta_Y_T_ini,gamma_Y_ini,test_Y,n,n1):

##use random forest to estimate U for test data
   
     rf_train = np.concatenate((treatment,M,X),axis=1)
     rf_label = Y_bias 
     
     model_y = clone(first_stage_reg().fit(rf_train, rf_label).best_estimator_)
     model_y.fit(rf_train,rf_label.reshape(n,))
     Y_bias_pred = model_y.predict(X = np.concatenate((test_treatment,test_M,test_X),axis=1)).reshape(n1,1)
    
     pred_Y = np.matmul(test_X[:,5:],beta_Y_ini) + np.matmul(test_M,beta_Y_M_ini) + test_treatment*beta_Y_T_ini  +  gamma_Y_ini*Y_bias_pred.reshape(n1,1)
     
     pred = (sum((pred_Y.reshape(n1,1) - test_Y.reshape(n1,1))**2)/n1)**0.5 
     return pred
 
    
  