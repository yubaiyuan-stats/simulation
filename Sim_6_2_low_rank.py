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
from src.DataGene_nonLinear_low_rank import *
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
      
      X_whole, Y_whole, treatment_whole, M_whole = DataGene_nonLinear_low_rank(N)
      
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
      Prop_FM(X= X, Y = Y, M = M, treatment = treatment,test_X=test_X,test_Y=test_Y,test_M=test_M,test_treatment=test_treatment,n=n,n1= n1,m=m, method ='low_rank',latent_dim = 3)
    
      mse_pca[j,ii] = pred
      PCA_treatment[j,ii] = h_PCA_treatment
      PCA_treatment_m[j,ii] = h_PCA_treatment_m
      PCA_treatment_d[j,ii] = h_PCA_Y_T
      
      ##proposed method using autoencoder as latent confounding model     
     
      h_autoencoder_treatment, h_autoencoder_treatment_m, h_autoencoder_Y_T, pred = \
      Prop_AE(X, Y, M, treatment,test_X,test_Y,test_M,test_treatment,Y_int,beta_Y_ini,beta_M_ini,beta_Y_M_ini,beta_Y_T_ini,T_M_ini,M_int,n,n1,m,method ='low_rank',latent_dim = 24,ini_dim = 6)

      mse_auto[j,ii] = pred
      autoencoder_treatment[j,ii] = h_autoencoder_treatment
      autoencoder_treatment_m[j,ii] = h_autoencoder_treatment_m
      autoencoder_treatment_d[j,ii] = h_autoencoder_Y_T 
      
## Causal Forest method

      fo_treatment, pred = Causal_Forest(X,M,Y,treatment,test_X,test_M,test_Y,n1,m,method ='low_rank')
      forest_treatment[j,ii] = fo_treatment
      mse_forest[j,ii] = pred 

## Xleaner method

      x_treatment, pred = X_learner(X,M,Y,treatment,test_X,test_M,test_Y,n,n1,m,method ='low_rank')
      X_treatment[j,ii] = x_treatment
      mse_X[j,ii] = pred 
 
##LSEM method

      lR_treatment,lR_treatment_m,lR_treatment_d,pred = LSEM(X,M,Y,treatment,test_X,test_M,test_Y,test_treatment,n1,m) 
      LR_treatment_m[j,ii] = lR_treatment_m
    #LR_treatment_d[j,ii] = lR_treatment_d
      LR_treatment[j,ii] = lR_treatment
      mse_lr[j,ii] = pred
    
 ###output the results in Table 2 under m = 5 with low-rank confounding effect
 ## the true treatment effect is 3, true mediation effect is 2.5 

#Prop FM
np.mean(abs(PCA_treatment-3),1) # value: 0.31, 0.14, 0.16
np.std(abs(PCA_treatment-3),1)  # value: 0.27, 0.13, 0.12
np.mean(abs(PCA_treatment_m-2.5),1)  # value: 1.01, 0.79, 0.47
np.std(abs(PCA_treatment_m-2.5),1)   # value: 0.78, 0.56, 0.42

np.mean(abs(mse_pca),1)  # value: 1.66, 1.64, 1.42
np.std(abs(mse_pca),1)   # value: 0.57, 0.27, 0.17

#Prop AE
np.mean(abs(autoencoder_treatment-3),1) # value: 0.86, 0.63, 0.57
np.std(abs(autoencoder_treatment-3),1)  # value: 0.31, 0.21, 0.14
np.mean(abs(autoencoder_treatment_m-2.5),1)  # value: 0.94, 0.79, 0.57
np.std(abs(autoencoder_treatment_m-2.5),1)   # value: 0.93, 0.51, 0.40

np.mean(abs(mse_auto),1)  # value: 1.84, 1.64, 1.44
np.std(abs(mse_auto),1)   # value: 0.56, 0.27, 0.17

#LSEM
np.mean(abs(LR_treatment-3),1)  # value: 1.09, 1.11, 1.10
np.std(abs(LR_treatment-3),1)   # value: 0.26, 0.12, 0.06
np.mean(abs(LR_treatment_m-2.5),1)  # value: 3.55, 3.62, 3.63
np.std(abs(LR_treatment_m-2.5),1)   # value: 0.44, 0.18, 0.13

np.mean(abs(mse_lr),1)  # value: 2.53, 2.05, 2.03
np.std(abs(mse_lr),1)   # value: 0.66, 0.23, 0.13

#Causal Forest
np.mean(forest_treatment,1)  # value: 4.49, 4.81, 5.03
np.std(forest_treatment,1)   # value: 0.91, 0.45, 0.53
np.mean(mse_forest,1)  # value: 2.23, 1.92, 1.75
np.std(mse_forest,1)   # value: 0.50, 0.24, 0.15

#XLearner
np.mean(X_treatment,1)  # value: 4.16, 4.67, 4.85
np.std(X_treatment,1)   # value: 0.86  0.57, 0.45
np.mean(mse_X,1)  # value: 2.19, 1.88, 1.72
np.std(mse_X,1)   # value: 0.07, 0.22, 0.16
      
      
####HIMA method
############# Import HIMA enviroment 

from src.HIMA_env import * 

rpy2.robjects.numpy2ri.activate()
importr('HIMA')   
    
repeat = 30      
size_num = np.array([200,800,2000]) 

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
      
      X_whole, Y_whole, treatment_whole, M_whole = DataGene_nonLinear_low_rank(N)
      
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
 
#HIMA      
np.mean(abs(hima_treatment-3),1)  # value: 1.82, 1.75, 1.79
np.std(abs(hima_treatment-3),1)   # value: 0.46, 0.20, 0.14
np.mean(abs(hima_treatment_m-2.5),1)  # value: 2.39, 2.82, 2.77
np.std(abs(hima_treatment_m-2.5),1)   # value: 0.86, 0.30, 0.14
np.mean(abs(hima_MSE),1)  # value: 2.53, 2.19, 2.27
np.std(abs(hima_MSE),1)   # value: 0.66, 0.23, 0.17
      
      
      
      
      
      
      

