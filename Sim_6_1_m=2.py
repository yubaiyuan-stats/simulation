#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from src.utli import *
from src.Prop_FM import * 
from src.Prop_AE import * 
from src.Causal_Forest import * 
from src.X_learner import * 
from src.LSEM import * 
from src.DataGene_Linear_k_2 import *
import numpy as np

sess = tf.InteractiveSession()     
    
#########simulation settup and record results
#with 30 replications at three different sample sizes
repeat = 30
size_num = np.array([200,800,2000])

##Prop AE     
autoencoder_treatment = np.zeros([3,repeat])

autoencoder_treatment_m = np.zeros([3,repeat])
mse_auto = np.zeros([3,repeat])
 
##Prop LF
PCA_treatment = np.zeros([3,repeat])
PCA_treatment_m = np.zeros([3,repeat])
mse_pca = np.zeros([3,repeat])

##SEM
LR_treatment = np.zeros([3,repeat])
LR_treatment_m = np.zeros([3,repeat])
mse_lr = np.zeros([3,repeat])

##Causal Forest
forest_treatment = np.zeros([3,repeat])
mse_forest = np.zeros([3,repeat])

##X Learner
X_treatment = np.zeros([3,repeat])
mse_X = np.zeros([3,repeat])


##start the simulation
for ii in range(repeat):
  
#fix the random seed for each run   
  np.random.seed(ii)  
  for j in range(3):
      
    N = size_num[j]
    
    X_whole, Y_whole, treatment_whole, M_whole = DataGene_Linear_k_2(N)
  
#split data into training and testing set randomly   

    test_ID = np.random.choice(range(N),int((0.2*N)),replace = False)
    train_ID = np.setdiff1d(np.array(range(N)), test_ID) 
    
    X = X_whole[train_ID,:]
    test_X = X_whole[test_ID,:]
    
    Y = Y_whole[train_ID] 
    test_Y =Y_whole[test_ID] 
    
    treatment = treatment_whole[train_ID] 
    test_treatment = treatment_whole[test_ID] 
    
    M = M_whole[train_ID,:] 
    test_M = M_whole[test_ID,:] 
    
    n = len(train_ID)
    n1 = len(test_ID)
    m = 2
    
###proposed method using factor model as latent confounding model    
  
    h_PCA_treatment, h_PCA_treatment_m, h_PCA_Y_T, pred, T_M_ini, beta_Y_M_ini, beta_M_ini, beta_Y_ini, beta_Y_T_ini, gamma_Y_ini, Y_int, M_int = \
    Prop_FM(X= X, Y = Y, M = M, treatment = treatment,test_X=test_X,test_Y=test_Y,test_M=test_M,test_treatment=test_treatment,n=n,n1= n1,m=m,method ='linear',latent_dim = 2)
    
    mse_pca[j,ii] = pred
    PCA_treatment[j,ii] = h_PCA_treatment
    PCA_treatment_m[j,ii] = h_PCA_treatment_m
    PCA_treatment_d[j,ii] = h_PCA_Y_T
    
##proposed method using autoencoder as latent confounding model     
   
    h_autoencoder_treatment, h_autoencoder_treatment_m, h_autoencoder_Y_T, pred = \
    Prop_AE(X, Y, M, treatment,test_X,test_Y,test_M,test_treatment,Y_int,beta_Y_ini,beta_M_ini,beta_Y_M_ini,beta_Y_T_ini,T_M_ini,M_int,n,n1,m,method ='linear',latent_dim = 2)

    mse_auto[j,ii] = pred
    autoencoder_treatment[j,ii] = h_autoencoder_treatment
    autoencoder_treatment_m[j,ii] = h_autoencoder_treatment_m
    autoencoder_treatment_d[j,ii] = h_autoencoder_Y_T   
    
## Causal Forest method

    fo_treatment, pred = Causal_Forest(X,M,Y,treatment,test_X,test_M,test_Y,n1,m,method ='linear')
    forest_treatment[j,ii] = fo_treatment
    mse_forest[j,ii] = pred 

## Xleaner method

    x_treatment, pred = X_learner(X,M,Y,treatment,test_X,test_M,test_Y,n,n1,m,method ='linear')
    X_treatment[j,ii] = x_treatment
    mse_X[j,ii] = pred 
 
##LSEM method

    lR_treatment,lR_treatment_m,lR_treatment_d,pred  = LSEM(X,M,Y,treatment,test_X,test_M,test_Y,test_treatment,n1,m) 
    LR_treatment_m[j,ii] = lR_treatment_m
    LR_treatment[j,ii] = lR_treatment
    mse_lr[j,ii] = pred
    
 ###output the results in Table 1 under m = 2
 ## the true treatment effect is 2, true mediation effect is 1 

#Prop FM
np.mean(abs(PCA_treatment-2),1) # value: 0.35, 0.18, 0.09
np.std(abs(PCA_treatment-2),1)  # value: 0.21, 0.15, 0.09
np.mean(abs(PCA_treatment_m-1),1) # value: 0.26, 0.12, 0.12
np.std(abs(PCA_treatment_m-1),1)  # value: 0.18, 0.08, 0.10
np.mean(abs(mse_pca),1) # value: 0.89
np.std(abs(mse_pca),1)  # value: 0.10
   
#Prop AE
np.mean(abs(autoencoder_treatment-2),1) # value: 0.53, 0.51, 0.43
np.std(abs(autoencoder_treatment-2),1)  # value: 0.18, 0.12, 0.10
np.mean(abs(autoencoder_treatment_m-1),1) # value: 0.33, 0.14, 0.13
np.std(abs(autoencoder_treatment_m-1),1)  # value: 0.21, 0.13, 0.08
np.mean(abs(mse_auto),1)  # value: 0.89, 0.86, 0.87
np.std(abs(mse_auto),1)   # value: 0.11, 0.04, 0.02
   
#LSEM
np.mean(abs(LR_treatment-2),1)  # value: 0.64, 0.66, 0.64
np.std(abs(LR_treatment-2),1)   # value: 0.13, 0.05, 0.04
np.mean(abs(LR_treatment_m-1),1)  # value: 1.14, 1.15, 1.13
np.std(abs(LR_treatment_m-1),1)   # value: 0.13, 0.06, 0.04
np.mean(abs(mse_lr),1)  # value: 0.96, 0.93, 0.93
np.std(abs(mse_lr),1)   # value: 0.10, 0.04, 0.03
   
#Causal Forest
np.mean(forest_treatment,1)  # value: 1.41, 1.48, 1.46
np.std(forest_treatment,1)   # value: 0.28, 0.11, 0.08
np.mean(mse_forest,1)  # value: 1.13, 1.00, 0.96
np.std(mse_forest,1)   # value: 0.15, 0.05, 0.03
   
#XLearner
np.mean(X_treatment,1)  # value: 1.36, 1.44, 1.45
np.std(X_treatment,1)   # value: 0.26, 0.12, 0.07
np.mean(mse_X,1)  # value: 1.11, 0.99, 0.96
np.std(mse_X,1)   # value: 0.13, 0.04, 0.03   
    

####HIMA method
############# Import HIMA enviroment 

from src.HIMA_env import * 

rpy2.robjects.numpy2ri.activate()
importr('HIMA') 

repeat = 30      
size_num = np.array([200,800,2000]) 

hima_treatment = np.zeros([3,repeat])
hima_treatment_m = np.zeros([3,repeat])

hima_MSE = np.zeros([3,repeat])

##start the simulation
for ii in range(repeat):
  
#fix the random seed for each run   
  np.random.seed(ii)  
  for j in range(3):
      
    N = size_num[j]
    
    X_whole, Y_whole, treatment_whole, M_whole = DataGene_Linear_k_2(N)
  
#split data into training and testing set randomly   

    test_ID = np.random.choice(range(N),int((0.2*N)),replace = False)
    train_ID = np.setdiff1d(np.array(range(N)), test_ID) 
    
    X = X_whole[train_ID,:]
    test_X = X_whole[test_ID,:]
    
    Y = Y_whole[train_ID] 
    test_Y =Y_whole[test_ID] 
    
    treatment = treatment_whole[train_ID] 
    test_treatment = treatment_whole[test_ID] 
    
    M = M_whole[train_ID,:] 
    test_M = M_whole[test_ID,:] 
    
    n = len(train_ID)
    n1 = len(test_ID)
    m = 2
## HIMA method

    Hima_treatment, Hima_treatment_m, Hima_MSE = HIMA(X,M,Y,treatment,test_X,test_M,test_Y,test_treatment,n1,m)
    hima_treatment_m[j,ii] = Hima_treatment_m
    hima_treatment[j,ii] = Hima_treatment
    hima_MSE[j,ii] = Hima_MSE

#HIMA
np.mean(abs(hima_treatment-2),1)  # value: 1.29, 1.26, 1.27
np.std(abs(hima_treatment-2),1)   # value: 0.24, 0.12, 0.07  
np.mean(abs(hima_treatment_m-1),1)  # value: 1.44, 1.38, 1.42
np.std(abs(hima_treatment_m-1),1)   # value: 0.21, 0.09, 0.06  
np.mean(abs(hima_MSE),1)  # value: 1.06, 1.04, 1.04  
np.std(abs(hima_MSE),1)   # value: 0.13, 0.07, 0.04




    
    
    

