#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

#import all the relevant packages and functions

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances
import scipy
import seaborn as sns
import copy
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.stats.stats import pearsonr

import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
from sklearn.linear_model import LogisticRegression

from scipy.linalg import hankel
import time

from numpy.linalg import inv
from sklearn.decomposition import PCA
import keras
from keras import layers

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from econml.sklearn_extensions.model_selection import GridSearchCVList
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone


from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner

from sklearn.datasets import make_circles
from keras import backend as K
from keras.constraints import UnitNorm, Constraint    
from sklearn.datasets import make_swiss_roll


#define functions used for proposed deconfounding algorithms and comopeting methods

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# grandient functions used in Prop FM and Prop AE
def para_grad(var,Loss_1,Loss_2,tune,ind):
    part_1 = 2*np.sum(-Loss_1*var)
    part_2 = 2*tune*(np.sum(Loss_1*Loss_2)/(LA.norm(Loss_1)*LA.norm(Loss_2)))*(np.sum(-var*Loss_2)/(LA.norm(Loss_1)*LA.norm(Loss_2)) -  (np.sum(Loss_1*Loss_2)*np.sum(-Loss_1*var))/(LA.norm(Loss_1)**3*LA.norm(Loss_2)))
    if ind:
       return (part_1+part_2)
    else:
       return ((1/2)*part_1+part_2)

def para_grad_vec(var,Loss_1,Loss_2,tune,ind):
    part_1 = -2*np.matmul(Loss_1.transpose(),var)
    part_2 = 2*tune*(np.sum(Loss_1*Loss_2)/(LA.norm(Loss_1)*LA.norm(Loss_2)))*(-np.matmul(Loss_2.transpose(),var)/(LA.norm(Loss_1)*LA.norm(Loss_2)) +  (np.sum(Loss_1*Loss_2)*np.matmul(Loss_1.transpose(),var))/(LA.norm(Loss_1)**3*LA.norm(Loss_2)))
    if ind:
       return (part_1+part_2)
    else:
       return ((1/2)*part_1+part_2)

## constratin functions in Prop AE
def corr_grad(Loss_M,gamma_U,n): 
    m = Loss_M.shape[1]
    corr_mat = np.corrcoef(Loss_M.transpose()) - np.diag(np.ones(m))
    grad_U = np.zeros(n)
    for i in range(Loss_M.shape[1]-1):
        for j in range((i+1),Loss_M.shape[1]):
            numrate = - gamma_U[i]*Loss_M[:,j] - gamma_U[j]*Loss_M[:,i] + np.sum(Loss_M[:,i]*Loss_M[:,j])*(gamma_U[i]*Loss_M[:,i]/LA.norm(Loss_M[:,i])**2 + gamma_U[j]*Loss_M[:,j]/LA.norm(Loss_M[:,j])**2)
            grad_U = grad_U + 2*corr_mat[i,j]*numrate/(LA.norm(Loss_M[:,i])*LA.norm(Loss_M[:,j]))
    return grad_U 

def corr_grad_T(Loss_M,treatment,P_est,gamma_U,gamma_t,n):   
    m = Loss_M.shape[1]
    corr_mat = np.corrcoef(Loss_M.transpose()) - np.diag(np.ones(m))
    grad_U = np.zeros(n)
    for i in range(Loss_M.shape[1]-1):
        for j in range((i+1),Loss_M.shape[1]):
            numrate = - gamma_U[i]*Loss_M[:,j] - gamma_U[j]*Loss_M[:,i] + np.sum(Loss_M[:,i]*Loss_M[:,j])*(gamma_U[i]*Loss_M[:,i]/LA.norm(Loss_M[:,i])**2 + gamma_U[j]*Loss_M[:,j]/LA.norm(Loss_M[:,j])**2)
            grad_U = grad_U + 2*corr_mat[i,j]*numrate/(LA.norm(Loss_M[:,i])*LA.norm(Loss_M[:,j]))
    
    T_res = (treatment - P_est).reshape(n,)
    corr_t = np.corrcoef(np.concatenate((T_res.reshape(n,1),Loss_M),axis=1).transpose())[0,1:]
    grad_U_t = np.zeros(n)
    for i in range(Loss_M.shape[1]):
        numrate = - gamma_U[i]*T_res - gamma_t*(P_est-P_est**2).reshape(n,)*Loss_M[:,i] + np.sum(Loss_M[:,i]*T_res)*(gamma_U[i]*Loss_M[:,i]/LA.norm(Loss_M[:,i])**2 + gamma_t*T_res*(P_est-P_est**2).reshape(n,)/LA.norm(T_res)**2)
        grad_U_t = grad_U_t + 2*corr_t[i]*numrate/(LA.norm(Loss_M[:,i])*LA.norm(T_res))

    return grad_U + grad_U_t 


## loss functions in Prop AE 
  

def treat_loss_fn(y_true,y_pred):
        y_pred_t = tf.clip_by_value(y_pred, clip_value_min=0.05, clip_value_max=0.95)
        treatment_loss =  -tf.reduce_sum(tf.log(y_pred_t)*y_true +  tf.log(1-y_pred_t)*(1-y_true)) - tf.reduce_sum(y_pred_t*tf.log(y_pred_t)) 
        return treatment_loss
    
    
    
def corr_loss_fn(Data):
        #m = T_M_Y.shape[1]
     def corr_fn(T_M_Y,T_M_Y_hat):   
        #m = T_M_Y.shape[1] 
        
        residual = T_M_Y - T_M_Y_hat
        n = residual.shape[1]
        corr_1 = tf.reduce_sum(tf.square(tfp.stats.correlation(residual)) - tf.linalg.diag(np.float32(np.ones(n))))
        corr_2 = tf.reduce_sum(tf.square(tf.linalg.tensor_diag_part(tfp.stats.correlation(residual,Data - residual))))
        return corr_1 + corr_2
     return corr_fn
    
def outcome_loss_fn(y_true,y_pred):
        a1 = tf.reduce_sum(tf.square(y_true[:,0] - y_pred[:,0]))
        a2 = tf.reduce_sum(tf.square(y_true[:,1:] - y_pred[:,1:]))
        return a1 + a2  
    
def norm_loss_fn(y_true,y_pred):
        a = tf.reduce_sum(tf.square(y_true - y_pred))
        return a
    
class UncorrelatedFeaturesConstraint_target (Constraint):
    
    def __init__(self, encoding_dim, target, weightage = 1.0 ):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.target = tf.cast(target, tf.float32)
    
    def get_covariance_target(self, x):
        corr_target_list = []

        for i in range(self.encoding_dim):
            corr_target_list.append(tf.math.abs(tfp.stats.correlation(x[:, i], self.target, sample_axis=0, event_axis=None)))
            
        corr_target = tf.stack(corr_target_list)
        total_corr_target = K.sum(K.square(corr_target))
        return total_corr_target
            

    def __call__(self, x):
        self.covariance = self.get_covariance_target(x)
        return self.weightage * self.covariance 
    
    
    
class correlatedFeaturesConstraint_target (Constraint):
    
    def __init__(self, encoding_dim, target, weightage = 1.0 ):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.target = tf.cast(target, tf.float32)
    
    def get_covariance_target_2(self, x):
        corr_target_list = []

        for i in range(self.encoding_dim):
            corr_target_list.append( 1 - tf.math.abs(tfp.stats.correlation(x[:, i], self.target, sample_axis=0, event_axis=None))  )
            
        corr_target = tf.stack(corr_target_list)
        total_corr_target = K.sum(K.square(corr_target))
        return total_corr_target
            

    def __call__(self, x):
        self.covariance = self.get_covariance_target_2(x)
        return self.weightage * self.covariance        
    
    
def my_ini(ini):    
   def my_init(shape, dtype=None):
     return tf.convert_to_tensor(ini) 
   return my_init


##functiuons related to Causal forest and XLeaner

def RBF_map(U,phi,order):
        feature_map = np.zeros([U.shape[0],order+1])
        scale = np.exp(-phi*(LA.norm(U,axis=1)**2).reshape(U.shape[0],1))
        a = 1
        if order > 1:
           a = (((2*phi)**order)/np.math.factorial(order))**0.5
        for i in range(order+1):
            j = order - i
            feature_map[:,i] = (a*scale*((U[:,0]**i)*(U[:,1]**j)).reshape(U.shape[0],1)).reshape(U.shape[0],)
        return feature_map        
    
 
def first_stage_reg_1():
    return GridSearchCVList([
                              RandomForestRegressor(n_estimators=30, random_state=123),
                              GradientBoostingRegressor(random_state=123)],
                              param_grid_list=[
                                                {'max_depth': [3, None],
                                                'min_samples_leaf': [10, 50]},
                                              {'n_estimators': [50, 100],
                                                'max_depth': [3,10],
                                                'min_samples_leaf': [10, 30]}],
                              cv=5,
                              scoring='neg_mean_squared_error')

def first_stage_reg():
    return GridSearchCVList([Lasso(),
                              RandomForestRegressor(n_estimators=30, random_state=123),
                              GradientBoostingRegressor(random_state=123)],
                              param_grid_list=[{'alpha': [.001, .01, .1, 1, 10]},
                                                {'max_depth': [3, None],
                                                'min_samples_leaf': [10, 50]},
                                              {'n_estimators': [50, 100],
                                                'max_depth': [3,10],
                                                'min_samples_leaf': [10, 30]}],
                              cv=5,
                              scoring='neg_mean_squared_error')


def first_stage_clf():
    return GridSearchCVList([LogisticRegression()],
                             param_grid_list=[{'C': [0.01, .1, 1, 10, 100]}],
                             cv=5,
                             scoring='neg_mean_squared_error')    

##generate simulated data

def piecewise_fn(ind, cond, value):   
    m = ind.shape[0]
    n = len(cond)
    output = np.zeros(m)
    for i in range(n):
      output[cond[i]] = value[i]
    return output        
    
    
def rvs(dim=3):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H  
 
    
 
    
 
    
    