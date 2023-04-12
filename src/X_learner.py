#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import numpy as np

def X_learner(X,M,Y,treatment,test_X,test_M,test_Y,n,n1,m,method):
    
    covariate = np.concatenate((X,M),axis=1)
    X_learner_models = GradientBoostingRegressor(n_estimators=50, max_depth=4, min_samples_leaf=int(n/100))
    X_learner_propensity_model = RandomForestClassifier(n_estimators=100, max_depth=6, 
                                                min_samples_leaf=int(n/100))
    X_learner = XLearner(models=X_learner_models, propensity_model=X_learner_propensity_model)
    # Train X_learner
    X_learner.fit(Y, treatment, X=covariate)
    # Estimate treatment effects on test data
    X_te = X_learner.effect(covariate)
    
    X_learner_models.fit(covariate,Y)
    pred_covariate = np.concatenate((test_X,test_M),axis=1)  
    pred = (sum((X_learner_models.predict(pred_covariate).reshape(n1,1) - test_Y.reshape(n1,1))**2)/n1)**0.5 
    
    if (m == 2)&(method == 'linear'):
        
        X_treatment = np.mean(abs(pred - 2))
        
    elif (m == 5)&(method == 'linear'):
         
        X_treatment = np.mean(abs(pred - 3.5))
        
    else:    
        
        X_treatment = np.mean(abs(pred - 3))
        
    return X_treatment, pred


 