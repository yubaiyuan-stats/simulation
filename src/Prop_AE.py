#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from src.Loss_update import *
from src.AE import *
from src.Loss_update_AE import *
from src.utli import *
from src.M_para import *
from src.Y_para_AE import *
from src.U_initialize import *
from src.Out_sample_pred import *
from src.Out_sample_pred_1 import *
sess = tf.InteractiveSession()  

def Prop_AE(X, Y, M, treatment,test_X,test_Y,test_M,test_treatment,Y_int,beta_Y_ini,beta_M_ini,beta_Y_M_ini,beta_Y_T_ini,T_M_ini,M_int,n,n1,m,method,latent_dim,ini_dim):
    
## set up algorithm parameters

    lambda_1 = 300
    lambda_2 = 1
    step_size = 0.05
    k = ini_dim 
    k1 = latent_dim
    U_scale = 10
    
## initialize the surrogate confounder U 
    
    U_ini = U_initialize(M,Y,k,n,U_scale,False)

## initialize the loss function
     
    res_Y, res_M, Loss_total = Loss_update(U_ini,treatment,X,M,Y,Y_int,beta_Y_ini,beta_M_ini,beta_Y_M_ini,beta_Y_T_ini,T_M_ini,M_int,lambda_1,lambda_2,n,m,k)

##set up the gradient descent algorithm
    
    total_loss_1 = copy.copy(Loss_total)
    loss_ind = total_loss_1
    total_loss = 0
    iter = 1
    predict_track = []
    pred_1 = 0
    
    autoencoder, encoder = AE(treatment,X,M,Y,lambda_1,lambda_2,n,m)
    
    while (total_loss_1 - total_loss> 1 and iter<=6) or iter <=2: 
        
        total_loss_1 = copy.copy(total_loss)  
   
        ## recover residual of M and Y using autoencoder
        
        M_Y = np.float32(np.concatenate((res_Y,res_M),axis=1))
        X_tr = treatment.reshape(n,1)
        Bias_norm_orth = np.matmul((np.diag(np.ones([n,1]).reshape(n,)) - np.matmul(X_tr,np.matmul(inv(np.matmul(X_tr.transpose(),X_tr)),X_tr.transpose()))),M_Y)
        T_M_Y = np.float32(np.concatenate((treatment,Bias_norm_orth),axis=1))
       
        ## train autoencoder by fitting the updated residuals
       
        autoencoder.fit(M_Y, [treatment,Bias_norm_orth,T_M_Y],
                epochs= 2000,   #+ 100*math.floor(iter),
                batch_size=n,verbose = 0)
  
        ## update the surrogate confounder U from autoencoder 
        
        encoded_U = encoder.predict(M_Y)
        Uhat = encoded_U
        Uhat = Uhat*(U_scale/np.linalg.norm(Uhat, axis=0))
        
        ## update the loss from autoencoder 
        
        total_loss, T_M_Y_hat = Loss_update_AE(autoencoder,M_Y,treatment,res_Y,res_M,lambda_1,lambda_2,n,m,sess)
        
        ## update the parameters of mediator model 
        
        gamma_M, beta_M, T_M, M_int,res_M = M_para(Uhat,treatment,X,M,n,k1,m,True)
        
        ## update the parameters of outcome model 
        
        beta_Y, beta_Y_M, gamma_Y_bias, beta_Y_T, res_Y = Y_para_AE(treatment,X,M,Y,T_M_Y_hat,Y_int,n,m)
        
        
        if method == 'low_rank':
            
           pred1 = Out_sample_pred(treatment,M,X,Y_bias,test_X,test_M,test_treatment,beta_Y,beta_Y_M,beta_Y_T,gamma_Y_bias,test_Y,predict_track,n,n1,method == 'low_rank')
        
        if total_loss < loss_ind:
       
            
            h_autoencoder_treatment_m = np.sum(beta_Y_M.reshape(1,m)*T_M)
            h_autoencoder_Y_T = beta_Y_T 
            h_autoencoder_treatment = h_autoencoder_treatment_m + h_autoencoder_Y_T
            
            Y_bias = T_M_Y_hat[:,1].reshape(n,1)
            T_M_auto = T_M
            beta_Y_M_auto = beta_Y_M
            beta_M_auto = beta_M
            beta_Y_auto = beta_Y
            beta_Y_T_auto = beta_Y_T  
            gamma_Y_auto = gamma_Y_bias
           
            loss_ind  = total_loss
            
            if method == 'low_rank':
               pred_1 = copy.copy(pred1)
           
            
        
        iter = iter + 1
        #print(iter)
        
    ## make the out-sample prediction for outcome based on random forest        
        
    pred = Out_sample_pred_1(treatment,M,X,Y_bias,test_X,test_M,beta_M,test_treatment,T_M_auto,beta_Y_auto,beta_Y_M_auto,beta_Y_T_auto,gamma_Y_auto,test_Y,n,n1) 
    pred_2 = Out_sample_pred(treatment,M,X,Y_bias,test_X,test_M,test_treatment,beta_Y_auto,beta_Y_M_auto,beta_Y_T_auto,gamma_Y_auto,test_Y,predict_track,n,n1,method == 'full_rank')
    
    if method == 'linear':
        
        Pred = pred
        
    elif method == 'low_rank':
        
        Pred = pred_1
        
    else: 
        
        Pred = pred_2
    
    return h_autoencoder_treatment, h_autoencoder_treatment_m, h_autoencoder_Y_T, Pred
    
               
       
            
                    

        
    
    
    


