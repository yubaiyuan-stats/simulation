#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import numpy as np
from sklearn.linear_model import LinearRegression
from numpy import linalg as LA

def Loss_sum(Uhat,treatment,X,M,Y,Y_int,beta_Y,beta_Y_M,beta_Y_T,M_int,beta_M,T_M,P_est,lambda_1,lambda_2,n,m,indi):
    
    U = copy.copy(Uhat)
    
    if indi == 'ini':
        
       

        Loss_Y = Y - (Y_int + np.matmul(X[:,5:],beta_Y).reshape(n,1) + np.matmul(M,beta_Y_M).reshape(n,1) + treatment*beta_Y_T)
        Loss_Y_center_U =  Loss_Y - np.mean(Loss_Y,axis = 0)   
        reg_Y_bias = LinearRegression(fit_intercept=False).fit(U, Loss_Y_center_U)
        gamma_Y = reg_Y_bias.coef_.transpose() 
        
        Loss_M = M - (M_int + np.matmul(X[:,0:5],beta_M.transpose()).reshape(n,m) + T_M*treatment)
        Loss_M_center_U =  Loss_M - np.mean(Loss_M,axis = 0)  
       

    else:
        
        Loss_Y = Y - (np.matmul(X[:,5:],beta_Y).reshape(n,1) + np.matmul(M,beta_Y_M).reshape(n,1) + treatment*beta_Y_T)
        reg_Y_bias = LinearRegression(fit_intercept=False).fit(U, Loss_Y)
        gamma_Y = reg_Y_bias.coef_.transpose() 
        Loss_Y_U = Y - (np.matmul(X[:,5:],beta_Y).reshape(n,1) + np.matmul(M,beta_Y_M).reshape(n,1) + treatment*beta_Y_T + np.matmul(U, gamma_Y))
        Loss_Y_center_U =  Loss_Y_U - np.mean(Loss_Y_U,axis = 0)   
        

        
        X_M_U = np.concatenate((U,X[:,0:5],treatment),axis=1)
        reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U, M)
        Loss_M_U = M - reg_M_U.predict(X_M_U)
        Loss_M_pred = M - (np.matmul(X[:,0:5],beta_M.transpose()).reshape(n,m) +  treatment*T_M)
        Loss_M_center_U =  Loss_M_U - np.mean(Loss_M_U,axis = 0) 
        
    Loss_M_T = np.concatenate((treatment - P_est,Loss_Y_center_U,Loss_M_center_U),axis=1)
        
    corr_res = LA.norm(np.corrcoef(Loss_M_T.transpose()) - np.diag(np.ones(m+2)))**2
    loss_T = -np.sum(np.log(P_est)*treatment + np.log(1-P_est)*(1-treatment)) #- np.sum(P_est*np.log(P_est))
    total_loss = np.sum(Loss_Y_center_U**2) + np.sum(Loss_M_center_U**2) + lambda_1*corr_res + lambda_2*loss_T
    
    
    if indi == 'ini':

       return total_loss,Loss_Y_center_U,Loss_M_center_U,gamma_Y
   
    else:
        
       return total_loss,Loss_Y_center_U,Loss_M_center_U, Loss_M_pred,gamma_Y
   
    
   
   