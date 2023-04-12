#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from econml.dml import CausalForestDML
from sklearn.base import clone
from econml.sklearn_extensions.model_selection import GridSearchCVList
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def first_stage_reg_1():
    return GridSearchCVList([
                              RandomForestRegressor(n_estimators=30, random_state=123),
                              GradientBoostingRegressor(random_state=123)],
                              param_grid_list=[
                                                {'max_depth': [3, None],
                                                'min_samples_leaf': [10, 50]},
                                              {'n_estimators': [50, 100],
                                                'max_depth': [3,10],
                                                'min_samples_leaf': [10, 30]}],
                              cv=5,
                              scoring='neg_mean_squared_error')

def first_stage_clf():
    return GridSearchCVList([LogisticRegression()],
                             param_grid_list=[{'C': [0.01, .1, 1, 10, 100]}],
                             cv=5,
                             scoring='neg_mean_squared_error')  


def Causal_Forest(X,M,Y,treatment,test_X,test_M,test_Y,n1,m,method):

    covariate = np.concatenate((X,M),axis=1)
    
    # #covariate = M
    model_y = clone(first_stage_reg_1().fit(covariate, Y).best_estimator_)
    
    model_t = clone(first_stage_clf().fit(covariate, treatment).best_estimator_)
    
    est = CausalForestDML(model_y=model_y,
                      model_t=model_t,
                      discrete_treatment=True,
                      cv=3,
                      n_estimators=20,
                      random_state=123)
    est.fit(Y, treatment, X=covariate)
    pred = est.effect(X = covariate)
    
    pred_covariate = np.concatenate((test_X,test_M),axis=1)  
    pred_1 = (sum((est.models_y[0][0].predict(X = pred_covariate).reshape(n1,1) - test_Y.reshape(n1,1))**2)/n1)**0.5 
    
    if (m == 2)&(method == 'linear'):
        
        forest_treatment = np.mean(abs(pred - 2))
        
    elif (m == 5)&(method == 'linear'):
         
        forest_treatment = np.mean(abs(pred - 3.5))
        
    else:    
        
        forest_treatment = np.mean(abs(pred - 3))
    
    return forest_treatment, pred_1




