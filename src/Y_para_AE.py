#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.linear_model import LinearRegression


def Y_para_AE(treatment,X,M,Y,T_M_Y_hat,Y_int,n,m):

    X_Y_U = np.concatenate((treatment,T_M_Y_hat[:,1].reshape(n,1), X[:,5:], M),axis=1)
    
    reg_Y_U = LinearRegression(fit_intercept=False).fit(X_Y_U, Y)
    beta_Y = reg_Y_U.coef_[0,2:(4)].reshape(2,1)
    beta_Y_M = reg_Y_U.coef_[0,(4):].reshape(m,1)
    #Y_int = reg_Y_U.intercept_
    gamma_Y_bias = reg_Y_U.coef_[0,1]
    beta_Y_T = reg_Y_U.coef_[0,0]
    
    res_Y = Y - (Y_int + np.matmul(X[:,5:],beta_Y).reshape(n,1) + np.matmul(M,beta_Y_M).reshape(n,1) + treatment*beta_Y_T)

    return beta_Y, beta_Y_M, gamma_Y_bias, beta_Y_T, res_Y
    
