#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
from src.utli import *
from src.M_para import *
from src.grad_U import *
from src.Loss_sum import *
from src.Out_sample_pred_1 import *
from src.Out_sample_pred import *
from src.T_para import *
from src.U_initialize import *
from src.Y_para import *


def Prop_FM(X, Y, M, treatment,test_X,test_Y,test_M,test_treatment,n,n1,m,method,latent_dim = 2):
    
## set up algorithm parameters

    lambda_1 = 2000
    lambda_2 = 1
    step_size = 0.001    
    k = latent_dim 
    U_scale = 10
    
## initialize the surrogate confounder U   
  
    U_ini = U_initialize(M,Y,k,n,U_scale,True)
    Uhat = copy.copy(U_ini)
   
## initialize the parameters in mediator model 
            
    gamma_M, beta_M, T_M, M_int = M_para(U_ini,treatment,X,M,n,k,m,False)
  
## initialize the parameters in outcoome model     
    
    X_Y_U = np.concatenate((treatment,U_ini,X[:,5:],M),axis=1)
    reg_Y_U = LinearRegression(fit_intercept=False).fit(X_Y_U, Y)
    beta_Y = reg_Y_U.coef_[0,(k+1):(k+3)].reshape(2,1)
    beta_Y_M = reg_Y_U.coef_[0,(k+3):].reshape(m,1)
    Y_int = reg_Y_U.intercept_
    gamma_Y = reg_Y_U.coef_[0,1:(k+1)].reshape(k,1)
    beta_Y_T = reg_Y_U.coef_[0,0]
    
## initialize the parameters in treatment model     
    
    gamma_T = 0.1*np.ones(k).reshape(k,1)
    P_est = sigmoid(np.matmul(U_ini,gamma_T))

## initialize the loss function

    Loss_total,Loss_Y_center_U,Loss_M_center_U,gamma_Y = Loss_sum(U_ini,treatment,X,M,Y,Y_int,beta_Y,beta_Y_M,beta_Y_T,M_int,beta_M,T_M,P_est,lambda_1,lambda_2,n,m,'ini')
    
##set up the gradient descent algorithm
    
    total_loss = 0
    loss_ind = copy.copy(Loss_total)
    iter = 1
    predict_track = []
    pred_1 = 0
   
    if method =='linear':
        N_inter = 500
    elif method =='low_rank': 
        N_inter = 200
    else:     
        N_inter = 600
        
        
    while iter<=N_inter:
        
            total_loss_1 = copy.copy(total_loss)
            
## update surrogate confounder by gradient descent 

            Uhat = np.matmul((np.diag(np.ones([n,1]).reshape(n,)) - np.matmul(X,np.matmul(inv(np.matmul(X.transpose(),X)),X.transpose()))),Uhat)
            grad_U_k = grad_U(n,k,gamma_M,gamma_T,gamma_Y,Loss_Y_center_U,Loss_M_center_U,treatment,P_est,lambda_1,lambda_2)
            Uhat = Uhat - step_size*grad_U_k
            Uhat = Uhat*(U_scale/np.linalg.norm(Uhat, axis=0))
            
## update the parameters in mediator model 
            
            gamma_M, beta_M, T_M, M_int = M_para(Uhat,treatment,X,M,n,k,m,False)
       
## update the parameters in outcome model 
    
            beta_Y, beta_Y_M, Y_int, gamma_Y_bias, beta_Y_T = Y_para(Uhat,treatment,X,M,Y,gamma_Y,m)
            Y_bias = np.matmul(Uhat,gamma_Y)
            
## update the parameters in treatment model            

            P_est, gamma_T = T_para(Uhat,treatment,n)
         
## update the loss function 
            
            total_loss,Loss_Y_center_U,Loss_M_center_U, Loss_M_pred,gamma_Y = Loss_sum(Uhat,treatment,X,M,Y,Y_int,beta_Y,beta_Y_M,beta_Y_T,M_int,beta_M,T_M,P_est,lambda_1,lambda_2,n,m,'update')

            if method == 'low_rank':
                
               pred1 = Out_sample_pred(treatment,M,X,Y_bias,test_X,test_M,test_treatment,beta_Y,beta_Y_M,beta_Y_T,gamma_Y_bias,test_Y,predict_track,n,n1,method == 'low_rank')
            
            if total_loss < loss_ind:
                
                h_PCA_treatment_m = np.sum(beta_Y_M.reshape(1,m)*T_M)
                h_PCA_Y_T = beta_Y_T 
                h_PCA_treatment = h_PCA_treatment_m + h_PCA_Y_T
                
                U_his = Uhat
                X_Y = np.concatenate((treatment,U_his,X[:,5:],M),axis=1)
                reg_Y = LinearRegression().fit(X_Y, Y)
                Y_bias = np.matmul(U_his,reg_Y.coef_[0,1:(k+1)].transpose()).reshape(n,1)
              
                loss_ind  = total_loss
                
                ##initial values for autoencoder algorithm
                
                T_M_ini = T_M
                beta_Y_M_ini = beta_Y_M
                beta_M_ini = beta_M
                beta_Y_ini = beta_Y
                beta_Y_T_ini = beta_Y_T         
                gamma_Y_ini = gamma_Y_bias
                
                if method == 'low_rank':
                   pred_1 = copy.copy(pred1)
                
            #print(iter)
            iter = iter + 1         
        
 ## make the out-sample prediction for outcome based on random forest        
      
    pred = Out_sample_pred_1(treatment,M,X,Y_bias,test_X,test_M,beta_M,test_treatment,T_M,beta_Y_ini,beta_Y_M_ini,beta_Y_T_ini,gamma_Y_ini,test_Y,n,n1)
    pred_2 = Out_sample_pred(treatment,M,X,Y_bias,test_X,test_M,test_treatment,beta_Y_ini,beta_Y_M_ini,beta_Y_T_ini,gamma_Y_ini,test_Y,predict_track,n,n1,method == 'full_rank')
    
    
    if method == 'linear':
        
        Pred = pred
        
    elif method == 'low_rank':
        
        Pred = pred_1
        
    else: 
        
        Pred = pred_2
            
    return h_PCA_treatment, h_PCA_treatment_m, h_PCA_Y_T, Pred, T_M_ini, beta_Y_M_ini, beta_M_ini, beta_Y_ini, beta_Y_T_ini, gamma_Y_ini, Y_int,M_int 
    

   
    
    
    
    