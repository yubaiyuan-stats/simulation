#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.utli import corr_grad_T
import numpy as np

def grad_U(n,k,gamma_M,gamma_T,gamma_Y,Loss_Y_center_U,Loss_M_center_U,treatment,P_est,lambda_1,lambda_2):

    grad_U_k = np.zeros([n,k])
    
    for kk in range(k):
    
      gamma_U = gamma_M[kk,:]
    
      gamma_t = gamma_T[kk,0]

    
      Loss_M_Y = np.concatenate((Loss_Y_center_U,Loss_M_center_U),axis=1)
      gamma_U_Y = np.concatenate((gamma_Y[kk,:],gamma_U))
    
    
      grad_corr = corr_grad_T(Loss_M_Y,treatment,P_est,gamma_U_Y,gamma_t, n)   
    
      grad_U_k[:,kk] = ( - 2*gamma_Y[kk,:]*Loss_Y_center_U  - np.sum(2*gamma_M[kk,:]*Loss_M_center_U,1).reshape(n,1) + lambda_1*grad_corr.reshape(n,1) + lambda_2*(-gamma_T[kk,0]*(treatment - P_est)).reshape(n,1)).reshape(n,)

    return grad_U_k

