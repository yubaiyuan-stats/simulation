#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression

def LSEM(X,M,Y,treatment,test_X,test_M,test_Y,test_treatment,n1,m):


    X_M_U = np.concatenate((X[:,0:5],treatment),axis=1)
    reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U, M)
    T_M =  reg_M_U.coef_[:,-1]
    
    
    
    X_Y_U = np.concatenate((treatment,M,X[:,5:]),axis=1)
    reg_Y_U = LinearRegression(fit_intercept=False).fit(X_Y_U, Y)
    beta_Y_M = reg_Y_U.coef_[0,1:(m+1)].reshape(m,1)
    beta_Y_T = reg_Y_U.coef_[0,0]
    beta_Y = reg_Y_U.coef_[0,(m+1):].reshape(2,1)
    
    LR_treatment_m = np.sum(beta_Y_M.transpose()*T_M)
    LR_treatment_d = beta_Y_T
    LR_treatment = np.sum(beta_Y_M.transpose()*T_M) + beta_Y_T
    pred_Y_SEM = np.matmul(test_X[:,5:],beta_Y) + np.matmul(test_M,beta_Y_M) + test_treatment*beta_Y_T
    pred = (sum((pred_Y_SEM.reshape(n1,1) - test_Y.reshape(n1,1))**2)/n1)**0.5 
    
    return LR_treatment,LR_treatment_m,LR_treatment_d,pred 
