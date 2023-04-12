#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression
import copy

def Y_para(Uhat,treatment,X,M,Y,gamma_Y,m):

    U_est = copy.copy(Uhat)
    Y_bias = np.matmul(U_est,gamma_Y)
    X_Y_U = np.concatenate((treatment, Y_bias, X[:,5:], M),axis=1)
    reg_Y_U = LinearRegression(fit_intercept=False).fit(X_Y_U, Y)
    beta_Y = reg_Y_U.coef_[0,2:(4)].reshape(2,1)
    beta_Y_M = reg_Y_U.coef_[0,(4):].reshape(m,1)
    Y_int = reg_Y_U.intercept_
    gamma_Y_bias = reg_Y_U.coef_[0,1]
    beta_Y_T = reg_Y_U.coef_[0,0]
    return beta_Y, beta_Y_M, Y_int, gamma_Y_bias, beta_Y_T



