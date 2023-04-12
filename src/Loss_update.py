#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy
from numpy import linalg as LA
def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def Loss_update(Uhat,treatment,X,M,Y,Y_int,beta_Y_ini,beta_M_ini,beta_Y_M_ini,beta_Y_T_ini,T_M_ini,M_int,lambda_1,lambda_2,n,m,k):
    
    n = n
    k = k
    m = m
    U = copy.copy(Uhat)
    gamma_T = 0.1*np.ones(k).reshape(k,1)
    T_int = 0
    res_Y = Y - (Y_int + np.matmul(X[:,5:],beta_Y_ini).reshape(n,1) + np.matmul(M,beta_Y_M_ini).reshape(n,1)+ treatment*beta_Y_T_ini)
    res_M = M - (M_int + np.matmul(X[:,0:5],beta_M_ini.transpose()).reshape(n,m) + T_M_ini*treatment)

    Loss_M_T = np.concatenate((treatment - sigmoid(np.matmul(U,gamma_T)+T_int),res_Y,res_M),axis=1)
    corr_res = LA.norm(np.corrcoef(Loss_M_T.transpose()) - np.diag(np.ones(m+2)))**2
    loss_T = -np.sum(np.log(sigmoid(np.matmul(U,gamma_T)+T_int))*treatment + np.log(1-sigmoid(np.matmul(U,gamma_T)+T_int))*(1-treatment)) 
    Loss_total = np.sum(res_Y**2) + np.sum(res_M**2) + lambda_1*corr_res + lambda_2*loss_T     
    return res_Y, res_M, Loss_total 




   