#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This is the code for reproducing the simulation resutls in Table 2
# for the low-rank nonlinear confounding effect setting 
# plesae first install following packages 
# "tensoeflow", "Keras", "econml", "rpy2" for competing methods

#import all the relevant modules   


from src.utli import *
from src.Prop_FM import * 
from src.Prop_AE import * 
from src.Causal_Forest import * 
from src.X_learner import * 
from src.LSEM import * 
from src.DataGene_nonLinear_full_rank import *
import numpy as np

sess = tf.InteractiveSession()     


#########simulation settup and record results
#with 30 replications at three different sample sizes
repeat = 30
size_num = np.array([1000,2000,3000])

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
      
      X_whole, Y_whole, treatment_whole, M_whole = DataGene_nonLinear_full_rank(N)
      
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
      m = 5
      
      ###proposed method using factor model as latent confounding model    
        
      h_PCA_treatment, h_PCA_treatment_m, h_PCA_Y_T, pred, T_M_ini, beta_Y_M_ini, beta_M_ini, beta_Y_ini, beta_Y_T_ini, gamma_Y_ini, Y_int, M_int = \
      Prop_FM(X= X, Y = Y, M = M, treatment = treatment,test_X=test_X,test_Y=test_Y,test_M=test_M,test_treatment=test_treatment,n=n,n1= n1,m=m, method ='full_rank',latent_dim = 3)
    
      mse_pca[j,ii] = pred
      PCA_treatment[j,ii] = h_PCA_treatment
      PCA_treatment_m[j,ii] = h_PCA_treatment_m
      PCA_treatment_d[j,ii] = h_PCA_Y_T
      
      ##proposed method using autoencoder as latent confounding model     
     
      h_autoencoder_treatment, h_autoencoder_treatment_m, h_autoencoder_Y_T, pred = \
      Prop_AE(X, Y, M, treatment,test_X,test_Y,test_M,test_treatment,Y_int,beta_Y_ini,beta_M_ini,beta_Y_M_ini,beta_Y_T_ini,T_M_ini,M_int,n,n1,m,method ='full_rank',latent_dim = 24,ini_dim = 6)

      mse_auto[j,ii] = pred
      autoencoder_treatment[j,ii] = h_autoencoder_treatment
      autoencoder_treatment_m[j,ii] = h_autoencoder_treatment_m
      autoencoder_treatment_d[j,ii] = h_autoencoder_Y_T 
      
## Causal Forest method

      fo_treatment, pred = Causal_Forest(X,M,Y,treatment,test_X,test_M,test_Y,n1,m,method ='full_rank')
      forest_treatment[j,ii] = fo_treatment
      mse_forest[j,ii] = pred 

## Xleaner method

      x_treatment, pred = X_learner(X,M,Y,treatment,test_X,test_M,test_Y,n,n1,m,method ='full_rank')
      X_treatment[j,ii] = x_treatment
      mse_X[j,ii] = pred 
 
##LSEM method

      lR_treatment,lR_treatment_m,lR_treatment_d,pred = LSEM(X,M,Y,treatment,test_X,test_M,test_Y,test_treatment,n1,m) 
      LR_treatment_m[j,ii] = lR_treatment_m
    #LR_treatment_d[j,ii] = lR_treatment_d
      LR_treatment[j,ii] = lR_treatment
      mse_lr[j,ii] = pred
    
###results in Table 2 under m = 5 under the full-rank confounding effect
## the true treatment effect is 3, true mediation effect is 2.5

#Prop FM
np.mean(abs(PCA_treatment-3),1)  # value: 1.29, 1.89, 1.94
np.std(abs(PCA_treatment-3),1)   # value: 0.60, 1.20, 1.11
np.mean(abs(PCA_treatment_m-2.5),1)  # value: 1.25, 1.93, 1.83
np.std(abs(PCA_treatment_m-2.5),1)   # value: 0.43, 0.86, 0.86

np.mean(abs(mse_pca),1)  # value: 2.23, 2.19, 2.18
np.std(abs(mse_pca),1)   # value: 0.11, 0.09, 0.07

#Prop AE
np.mean(abs(autoencoder_treatment-3),1)  # value: 1.08, 1.34, 1.52
np.std(abs(autoencoder_treatment-3),1)   # value: 0.52, 0.73, 0.91
np.mean(abs(autoencoder_treatment_m-2.5),1)  # value: 0.97, 1.37, 1.59
np.std(abs(autoencoder_treatment_m-2.5),1)   # value: 0.72, 0.82, 0.66

np.mean(abs(mse_auto),1)  # value: 2.28, 2.21, 2.21
np.std(abs(mse_auto),1)   # value: 0.11, 0.09, 0.06

#LSEM
np.mean(abs(LR_treatment-3),1)  # value: 2.03, 2.01, 1.96
np.std(abs(LR_treatment-3),1)   # value: 0.14, 0.11, 0.08
np.mean(abs(LR_treatment_m-2.5),1)  # value: 3.50, 3.62, 3.54
np.std(abs(LR_treatment_m-2.5),1)   # value: 0.23, 0.13, 0.17

np.mean(abs(mse_lr),1)  # value: 2.59, 2.56, 2.56
np.std(abs(mse_lr),1)   # value: 0.10, 0.05, 0.06

#Causal Forest 
np.mean(forest_treatment,1)  # value: 5.47, 5.13, 4.82
np.std(forest_treatment,1)   # value: 0.88, 0.64, 0.55
np.mean(mse_forest,1)  # value: 2.42, 2.32, 2.28
np.std(mse_forest,1)   # value: 0.08, 0.09, 0.06

#XLearner
np.mean(X_treatment,1)  # value: 4.14, 4.25, 4.12
np.std(X_treatment,1)   # value: 0.52, 0.35, 0.35
np.mean(mse_X,1)  # value: 2.38, 2.30, 2.27
np.std(mse_X,1)   # value: 0.09, 0.08, 0.06
      
      
####HIMA method
############# Import HIMA enviroment 

from src.HIMA_env import * 

rpy2.robjects.numpy2ri.activate()
importr('HIMA') 
     
repeat = 30      
size_num = np.array([1000,2000,3000]) 

hima_treatment = np.zeros([3,repeat])
hima_treatment_m = np.zeros([3,repeat])
#hima_treatment_d =np.zeros([3,repeat])
hima_MSE = np.zeros([3,repeat])

##start the simulation
for ii in range(repeat):
  
#fix the random seed for each run   
  np.random.seed(ii)  
  for j in range(3):
     
      N = size_num[j]
      
      X_whole, Y_whole, treatment_whole, M_whole = DataGene_nonLinear_full_rank(N)
      
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
      m = 5
      
      ## HIMA method

      Hima_treatment, Hima_treatment_m, Hima_MSE = HIMA(X,M,Y,treatment,test_X,test_M,test_Y,test_treatment,n1,m)
      hima_treatment_m[j,ii] = Hima_treatment_m
      hima_treatment[j,ii] = Hima_treatment
      hima_MSE[j,ii] = Hima_MSE
   
# HIMA      
np.mean(abs(hima_treatment-3),1)  # value: 4.07, 3.96, 4.01
np.std(abs(hima_treatment-3),1)   # value: 0.22, 0.14, 0.14
np.mean(abs(hima_treatment_m-2.5),1)  # value: 4.58, 4.48, 4.56
np.std(abs(hima_treatment_m-2.5),1)   # value: 0.72, 0.34, 0.26
np.mean(abs(hima_MSE),1)  # value: 2.65, 2.62, 2.63
np.std(abs(hima_MSE),1)   # value: 0.18, 0.09, 0.08
      
      
      
      
      
      
      

