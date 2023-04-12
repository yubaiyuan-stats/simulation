#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
from sklearn.linear_model import LinearRegression
import copy

def M_para(Uhat,treatment,X,M,n,k,m,resi):

    
    U_est = copy.copy(Uhat)
    X_M_U = np.concatenate((U_est,X[:,0:5],treatment),axis=1)
    reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U, M)
    gamma_M = reg_M_U.coef_[:,0:k].transpose()
    beta_M = reg_M_U.coef_[:,k:(k+5)]
    T_M =  reg_M_U.coef_[:,-1]
    M_int = reg_M_U.intercept_
    if resi == False:
     return gamma_M, beta_M, T_M, M_int 
    else:
     res_M = M - (M_int + np.matmul(X[:,0:5],beta_M.transpose()).reshape(n,m) + T_M*treatment)   
     return gamma_M, beta_M, T_M, M_int,res_M 
 
    
 
   

