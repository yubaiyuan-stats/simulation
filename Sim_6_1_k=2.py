
"""
@author: Yubai Yuan
"""



# This is the code for reproducing the simulation resutls in Table 1
# when number of mediators is k = 2
# plesae first install following packages 
# "tensoeflow", "Keras", "econml", "rpy2" for competing methods

#import all the relevant modules   
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
def corr_grad(Loss_M,gamma_U): 
    m = Loss_M.shape[1]
    corr_mat = np.corrcoef(Loss_M.transpose()) - np.diag(np.ones(m))
    grad_U = np.zeros(n)
    for i in range(Loss_M.shape[1]-1):
        for j in range((i+1),Loss_M.shape[1]):
            numrate = - gamma_U[i]*Loss_M[:,j] - gamma_U[j]*Loss_M[:,i] + np.sum(Loss_M[:,i]*Loss_M[:,j])*(gamma_U[i]*Loss_M[:,i]/LA.norm(Loss_M[:,i])**2 + gamma_U[j]*Loss_M[:,j]/LA.norm(Loss_M[:,j])**2)
            grad_U = grad_U + 2*corr_mat[i,j]*numrate/(LA.norm(Loss_M[:,i])*LA.norm(Loss_M[:,j]))
    return grad_U 

def corr_grad_T(Loss_M,treatment,P_est,gamma_U,gamma_t):   
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
        corr_1 = tf.reduce_sum(tf.square(tfp.stats.correlation(residual)) - tf.linalg.diag(np.float32(np.ones(4))))
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
    
    
sess = tf.InteractiveSession()     
    
#########simulation settup and record results
#with 30 replications at three different sample sizes
repeat = 30
size_num = np.array([200,800,2000])
    
autoencoder_treatment = np.zeros([3,repeat])
autoencoder_treatment_d = np.zeros([3,repeat])
autoencoder_treatment_m = np.zeros([3,repeat])

autoencoder_T_bias = np.zeros([3,repeat])
autoencoder_Y_bias = np.zeros([3,repeat])

autoencoder_iter = np.zeros([3,repeat])
mse_auto = np.zeros([3,repeat])
 

PCA_treatment = np.zeros([3,repeat])
PCA_treatment_d = np.zeros([3,repeat])
PCA_treatment_m = np.zeros([3,repeat])
PCA_T_bias = np.zeros([3,repeat])
PCA_Y_bias = np.zeros([3,repeat])
mse_pca = np.zeros([3,repeat])


LR_treatment = np.zeros([3,repeat])
LR_treatment_d = np.zeros([3,repeat])
LR_treatment_m = np.zeros([3,repeat])
mse_lr = np.zeros([3,repeat])

 
LR_T_bias = np.zeros([3,repeat])
LR_Y_bias = np.zeros([3,repeat])


forest_treatment = np.zeros([3,repeat])
mse_forest = np.zeros([3,repeat])

X_treatment = np.zeros([3,repeat])
mse_X = np.zeros([3,repeat])

    
    
for ii in range(repeat):
  
#fix the random seed for each run   
  np.random.seed(ii)  
  for j in range(3):
      
    N = size_num[j]
    
##generate the underlying latent confounder U   
    U = np.zeros([N,1])
    U_ind = np.zeros([N,1])
    for i in range(N):
        ind =  np.random.binomial(1,0.5)
        U_ind[i] = ind
        if ind == 1:
           U[i] = np.random.normal(2,1.5,1)
        else: 
           U[i] = np.random.normal(-2,1.5,1)
           
#generate treatment assignment T        
    P_whole = np.exp(0.4*U)/(1+np.exp(0.4*U))
    treatment_whole = np.random.binomial(1,P_whole)
    
#generate X, M ,and Y       
    Bias=  [0.6,0.8,2]*U
    Bias = (Bias - np.mean(Bias,axis=0))/np.std(Bias,axis = 0)
   
    
    X_whole = np.random.multivariate_normal(np.zeros(7), np.diag(np.ones(7)), N)
    
    noise_M = np.random.multivariate_normal(np.zeros(2), 0.5*np.diag(np.ones(2)), N)
    coef_M = rvs(dim=5)
    M_whole = 0.5*Bias[:,0:2] + [1,1]*treatment_whole + 0.5*np.matmul(X_whole[:,0:5],coef_M[:,0:2]) + 1*noise_M
    true_M_whole = 0.5*Bias[:,0:2] + [1,1]*treatment_whole + 0.5*np.matmul(X_whole[:,0:5],coef_M[:,0:2])
        
    noise_Y = np.random.multivariate_normal([0], [[0.3]], N)
    Y_whole = 1*treatment_whole + (1*Bias[:,2]).reshape(N,1) + np.matmul(M_whole,[0.5,0.5]).reshape(N,1) + 0.3*np.matmul(X_whole[:,5:],[1,1]).reshape(N,1) + 1*noise_Y
    true_Y_whole = 1*treatment_whole + 1*Bias[:,2].reshape(N,1) + np.matmul(M_whole,[0.5,0.5]).reshape(N,1) + 0.3*np.matmul(X_whole[:,5:],[1,1]).reshape(N,1)
    np.corrcoef(np.concatenate((true_Y_whole,Bias[:,2].reshape(N,1)),axis=1).transpose())
    
#split data into training and testing set    
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
    
    P = P_whole[train_ID]
    test_P = P_whole[test_ID]
    
    n = len(train_ID)
    n1 = len(test_ID)
     
######proposed method using matrix factorization as latent confounding model    
    
##set dimension of surrogate confounder as k = 2    
    k = 2

    U_scale = 10
    
##obtain the initial of U    
    pca = PCA(n_components=k)
    
    
    
    pca.fit(np.concatenate((Y,M),axis=1))
    pca.explained_variance_ratio_
    
    U_ini = pca.fit_transform(np.concatenate((Y,M),axis=1)) #+ 2*np.random.randn(n, k) 
    
    reg_Uini = LinearRegression().fit(U_ini,P)
    corr_track = [np.corrcoef(np.concatenate((reg_Uini.predict(U_ini),P),axis=1).transpose())[0,1]]   
    
   
 
    U_ini = ((U_scale/np.linalg.norm(U_ini, axis=0))*U_ini).reshape(n,k)
    Uhat = copy.copy(U_ini)
   
##initialize the parameters    
    U_est = copy.copy(U_ini)
    X_M_U = np.concatenate((U_est,X[:,0:5],treatment),axis=1)
    reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U, M)
    gamma_M = reg_M_U.coef_[:,0:k].transpose()
    beta_M = reg_M_U.coef_[:,k:(k+5)]
    T_M =  reg_M_U.coef_[:,-1]
    M_int = reg_M_U.intercept_


    X_Y_U = np.concatenate((treatment,U_est,X[:,5:],M),axis=1)
    reg_Y_U = LinearRegression(fit_intercept=False).fit(X_Y_U, Y)
    beta_Y = reg_Y_U.coef_[0,(k+1):(k+3)].reshape(2,1)
    beta_Y_M = reg_Y_U.coef_[0,(k+3):].reshape(2,1)
    Y_int = reg_Y_U.intercept_
    gamma_Y = reg_Y_U.coef_[0,1:(k+1)].reshape(k,1)
    beta_Y_T = reg_Y_U.coef_[0,0]

    T_M_track = [np.sum(beta_Y_M.reshape(1,2)*T_M)]
    beta_Y_T_track = [beta_Y_T]
 

    
    U_begin = copy.copy(U_ini)

#set up the penalization parameters
    lambda_1 = 2000
    lambda_2 = 1
   
    
    Loss_Y = Y - (Y_int + np.matmul(X[:,5:],beta_Y).reshape(n,1) + np.matmul(M,beta_Y_M).reshape(n,1) + treatment*beta_Y_T)
    Loss_Y_center_U =  Loss_Y - np.mean(Loss_Y,axis = 0)   
    reg_Y_bias = LinearRegression(fit_intercept=False).fit(U_begin, Loss_Y_center_U)
    gamma_Y = reg_Y_bias.coef_.transpose()

    
    Loss_M = M - (M_int + np.matmul(X[:,0:5],beta_M.transpose()).reshape(n,2) + T_M*treatment)
    Loss_M_center_U =  Loss_M - np.mean(Loss_M,axis = 0)  
   
    gamma_T = 0.1*np.ones(k).reshape(k,1)
   
    T_int = 0
    P_est = sigmoid(np.matmul(U_begin,gamma_T)+T_int)
   
    
    Loss_M_T = np.concatenate((treatment - P_est,Loss_Y_center_U,Loss_M_center_U),axis=1)
    
    
    corr_res = LA.norm(np.corrcoef(Loss_M_T.transpose()) - np.diag(np.ones(4)))**2
    
  
    loss_T = -np.sum(np.log(P_est)*treatment + np.log(1-P_est)*(1-treatment)) #- np.sum(P_est*np.log(P_est))
    
    
    scale = 1/n
    
    Loss_total = np.sum(Loss_Y_center_U**2) + np.sum(Loss_M_center_U**2) + lambda_1*corr_res + lambda_2*loss_T #+ lambda_3*loss_L #+ lambda_4*loss_orth
    
    
#set up the step size of gradient descent 
       
    step_size = 0.001
    
    total_loss_1 = copy.copy(Loss_total)

    total_loss = 0
    iter = 1
    loss_track = [total_loss_1]
    loss_track_Y = [np.sum(Loss_Y**2)]
    loss_track_M = [np.sum(Loss_M**2)]
    loss_track_corr_res = [corr_res**2]
    loss_track_T = [loss_T]
    
    loss_ind = total_loss_1
    
    
    T_M_ini = 0 
    beta_Y_M_ini = 0
    beta_M_ini = 0
    beta_Y_ini = 0
    beta_Y_T_ini = 0
    
    h_PCA_treatment = 0
    h_PCA_treatment_m = 0
    h_PCA_MSE = 0
    h_PCA_U = 0
    h_PCA_P = 0
    h_PCA_P_2 = 0
    h_PCA_T_bias = 0
    h_PCA_Y_bias = 0
    h_PCA_Y_T = 0
    

    fit_track = []
    corr_rec_U = []
    predict_track = []

    
  
    while iter<=500:
        
            total_loss_1 = copy.copy(total_loss)
           
            Uhat = np.matmul((np.diag(np.ones([n,1]).reshape(n,)) - np.matmul(X,np.matmul(inv(np.matmul(X.transpose(),X)),X.transpose()))),Uhat)
                   
           
 ##update surrogate confounder          
            grad_U_k = np.zeros([n,k])
            
            for kk in range(k):
                
                gamma_U = gamma_M[kk,:]
                
                gamma_t = gamma_T[kk,0]
       
                
                Loss_M_Y = np.concatenate((Loss_Y_center_U,Loss_M_center_U),axis=1)
                gamma_U_Y = np.concatenate((gamma_Y[kk,:],gamma_U))
                
                grad_corr = corr_grad_T(Loss_M_Y,treatment,P_est,gamma_U_Y,gamma_t)   
                
                grad_U_k[:,kk] = ( - 2*gamma_Y[kk,:]*Loss_Y_center_U  - np.sum(2*gamma_M[kk,:]*Loss_M_center_U,1).reshape(n,1) + lambda_1*grad_corr.reshape(n,1) + lambda_2*(-gamma_T[kk,0]*(treatment - P_est)).reshape(n,1)).reshape(n,)

            
              
            Uhat = Uhat - step_size*grad_U_k
            Uhat = Uhat*(U_scale/np.linalg.norm(Uhat, axis=0))
            
## update the linear models for two mediators M and outcome Y 
            
            U_est = copy.copy(Uhat)
            X_M_U = np.concatenate((U_est,X[:,0:5],treatment),axis=1)
            reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U, M)
            gamma_M = reg_M_U.coef_[:,0:k].transpose()
            beta_M = reg_M_U.coef_[:,k:(k+5)]
            T_M =  reg_M_U.coef_[:,-1]
            M_int = reg_M_U.intercept_
       
    
            
            Y_bias = np.matmul(U_est,gamma_Y)
            X_Y_U = np.concatenate((treatment, Y_bias, X[:,5:], M),axis=1)
            reg_Y_U = LinearRegression(fit_intercept=False).fit(X_Y_U, Y)
            beta_Y = reg_Y_U.coef_[0,2:(4)].reshape(2,1)
            beta_Y_M = reg_Y_U.coef_[0,(4):].reshape(2,1)
            Y_int = reg_Y_U.intercept_
            gamma_Y_bias = reg_Y_U.coef_[0,1]
            beta_Y_T = reg_Y_U.coef_[0,0]
            
           
            fit_track.append((sum((Y.reshape(n,1) - reg_Y_U.predict(X_Y_U).reshape(n,1))**2)/n)**0.5) 
         
            treat_model = LogisticRegression(fit_intercept=False).fit(U_est, treatment.reshape(n,))
            P_est = treat_model.predict_proba(U_est)[:,1].reshape(n,1)
            gamma_T = treat_model.coef_.transpose()
            
            
##calculate the loss from model fitting
              
            Loss_Y = Y - (np.matmul(X[:,5:],beta_Y).reshape(n,1) + np.matmul(M,beta_Y_M).reshape(n,1) + treatment*beta_Y_T)
            reg_Y_bias = LinearRegression(fit_intercept=False).fit(U_est, Loss_Y)# - np.mean(Loss_Y))
            gamma_Y = reg_Y_bias.coef_.transpose()
            
            Loss_Y_U = Y - (np.matmul(X[:,5:],beta_Y).reshape(n,1) + np.matmul(M,beta_Y_M).reshape(n,1) + treatment*beta_Y_T + np.matmul(U_est, gamma_Y))
            Loss_Y_center_U =  Loss_Y_U - np.mean(Loss_Y_U,axis = 0)   

            
           
            Loss_M_U = M - reg_M_U.predict(X_M_U)
            Loss_M_center_U =  Loss_M_U - np.mean(Loss_M_U,axis = 0)  
            Loss_M_pred = M - (np.matmul(X[:,0:5],beta_M.transpose()).reshape(n,2) +  treatment*T_M)
            
            
            
            Loss_M_T = np.concatenate((treatment - P_est,Loss_Y_center_U,Loss_M_center_U),axis=1)
            corr_res = LA.norm(np.corrcoef(Loss_M_T.transpose()) - np.diag(np.ones(4)))**2
            loss_T = -np.sum(np.log(P_est)*treatment + np.log(1-P_est)*(1-treatment)) #- np.sum(P_est*np.log(P_est))

           
            total_loss = np.sum(Loss_Y_center_U**2) + np.sum(Loss_M_center_U**2) + lambda_1*corr_res + lambda_2*loss_T #+ lambda_3*loss_L #+ lambda_4*loss_orth
           
   
            iter = iter + 1
            
 
            loss_track.append(total_loss)
            T_M_track.append(np.sum(beta_Y_M.reshape(1,2)*T_M))
            beta_Y_T_track.append(beta_Y_T)
            
            
##make the out-sample prediction for outcome based on random forest  

            rf_train = np.concatenate((treatment,Loss_M_pred),axis=1)
            rf_label = Y_bias
    
            rf = RandomForestRegressor(n_estimators = 100, max_depth=12,random_state = 123)
            
         
            
            Loss_M_U_test = test_M - (np.matmul(test_X[:,0:5],beta_M.transpose()).reshape(n1,2) +  test_treatment*T_M)
            
            rf.fit(rf_train,rf_label.reshape(n,))
            
            Y_bias_pred = rf.predict(np.concatenate((test_treatment,Loss_M_U_test),axis=1)).reshape(n1,1)
            
            pred_Y = np.matmul(test_X[:,5:],beta_Y) + np.matmul(test_M,beta_Y_M) + test_treatment*beta_Y_T + gamma_Y_bias*Y_bias_pred.reshape(n1,1)
    
            predict_track.append((sum((pred_Y.reshape(n1,1) - test_Y.reshape(n1,1))**2)/n1)**0.5) 
            
            
            
            
            if total_loss < loss_ind:
                
                h_PCA_T_bias = np.mean(abs(T_M - np.ones([1,2])))     
                h_PCA_Y_bias = np.mean(abs(beta_Y_M - 0.5*np.ones([2,1]))) 
                h_PCA_treatment_m = np.sum(beta_Y_M.reshape(1,2)*T_M)
                #h_PCA_MSE = LA.norm(np.concatenate((Loss_Y - noise_Y, Loss_M - noise_M),axis=1))/n**0.5
    
                U_his = Uhat
                treat_model = LogisticRegression(random_state=0,fit_intercept=False).fit(U_his, treatment.reshape(n,))
                prop_score = treat_model.predict_proba(U_his)
                h_PCA_P = np.corrcoef(np.concatenate((prop_score[:,1].reshape(n,1),P),axis=1).transpose())[0,1]
                
                reg_Uhat = LinearRegression(fit_intercept=False).fit(U_his,P)
                h_PCA_P_2 =  np.corrcoef(np.concatenate((reg_Uhat.predict(U_his),P),axis=1).transpose())[0,1]
    
        
              
                X_M = np.concatenate((U_his,X[:,0:5],treatment),axis=1)
                reg_M = LinearRegression().fit(X_M, M)
                
                X_Y = np.concatenate((treatment,U_his,X[:,5:],M),axis=1)
                reg_Y = LinearRegression().fit(X_Y, Y)
                Y_bias = np.matmul(U_his,reg_Y.coef_[0,1:(k+1)].transpose()).reshape(n,1)
              
            
                h_PCA_Y_T = beta_Y_T 
                h_PCA_treatment = h_PCA_treatment_m + h_PCA_Y_T
                
               
                loss_ind  = total_loss
                
                T_M_ini = T_M
                beta_Y_M_ini = beta_Y_M
                beta_M_ini = beta_M
                beta_Y_ini = beta_Y
                beta_Y_T_ini = beta_Y_T        
                gamma_Y_ini = gamma_Y_bias
                
                mse_3_pca = min(predict_track)
            
            print(iter)
            
       
      
    rf_train = np.concatenate((treatment,M,X),axis=1)
    rf_label = Y_bias 
    
    model_y = clone(first_stage_reg().fit(rf_train, rf_label).best_estimator_)
    model_y.fit(rf_train,rf_label.reshape(n,))
    Y_bias_pred = model_y.predict(X = np.concatenate((test_treatment,test_M,test_X),axis=1)).reshape(n1,1)
 
    pred_Y = np.matmul(test_X[:,5:],beta_Y_ini) + np.matmul(test_M,beta_Y_M_ini) + test_treatment*beta_Y_T_ini  +  gamma_Y_ini*Y_bias_pred.reshape(n1,1)
    
    mse_pca[j,ii] = (sum((pred_Y.reshape(n1,1) - test_Y.reshape(n1,1))**2)/n1)**0.5 
 
    PCA_T_bias[j,ii] = h_PCA_T_bias  
    PCA_Y_bias[j,ii] = h_PCA_Y_bias
    PCA_treatment[j,ii] = h_PCA_treatment
    PCA_treatment_m[j,ii] = h_PCA_treatment_m
    PCA_treatment_d[j,ii] = h_PCA_Y_T
    
    
###proposed method using autoencoder as latent confounding model   

    k  = 2
    U_scale = 10   
    
    pca = PCA(n_components=2)
   
    pca.fit(np.concatenate((Y,M),axis=1))
    
      
    U_ini = pca.fit_transform(np.concatenate((Y,M),axis=1)) 
   
    reg_Uini = LinearRegression().fit(U_ini,P)
    U_begin = copy.copy(U_ini)
    
    
    corr_track = [0,np.corrcoef(np.concatenate((reg_Uini.predict(U_ini),P),axis=1).transpose())[0,1]] 
    gamma_T = 0.1*np.ones(k).reshape(k,1)
   
    T_int = 0
    
    
    
    lambda_1 = 300
    lambda_2 = 1

    res_Y = Y - (Y_int + np.matmul(X[:,5:],beta_Y_ini).reshape(n,1) + np.matmul(M,beta_Y_M_ini).reshape(n,1)+ treatment*beta_Y_T_ini)
    linear_Y = (Y_int + np.matmul(X[:,5:],beta_Y_ini).reshape(n,1) + np.matmul(M,beta_Y_M_ini).reshape(n,1) + treatment*beta_Y_T_ini)
    gamma_Y = 0
    res_M = M - (M_int + np.matmul(X[:,0:5],beta_M_ini.transpose()).reshape(n,2) + T_M_ini*treatment)
    linear_M = (M_int + np.matmul(X[:,0:5],beta_M_ini.transpose()).reshape(n,2) + T_M_ini*treatment)
    
    T_M_track = [np.sum(T_M_ini*beta_Y_M_ini.transpose())]
    Y_T_track = [beta_Y_T_ini]
 
    
    Loss_M_T = np.concatenate((treatment - sigmoid(np.matmul(U_begin,gamma_T)+T_int),res_Y,res_M),axis=1)
    
    corr_res = LA.norm(np.corrcoef(Loss_M_T.transpose()) - np.diag(np.ones(4)))**2
    
    P_est = sigmoid(np.matmul(U_begin,gamma_T)+T_int)
    
    loss_T = -np.sum(np.log(sigmoid(np.matmul(U_begin,gamma_T)+T_int))*treatment + np.log(1-sigmoid(np.matmul(U_begin,gamma_T)+T_int))*(1-treatment)) 
     
    
    Loss_total = np.sum(res_Y**2) + np.sum(res_M**2) + lambda_1*corr_res + lambda_2*loss_T #+ lambda_3*loss_L #+ lambda_4*loss_orth
    
   
    
    step_size = 0.05
    
    total_loss_1 = copy.copy(Loss_total)
    total_loss = 0
    iter = 1
    loss_track = [total_loss_1]
    outcome_track = [np.sum(res_Y**2) + np.sum(res_M**2)]
    
    loss_track_Y = [np.sum(res_Y**2)+np.sum(res_M**2)]
    
    loss_track_corr_res = [corr_res**2]
    loss_track_T = [loss_T]
    loss_norm = [0]
    
    
    loss_ind = total_loss_1
    corr_ind = corr_track[0]
    
    
    h_autoencoder_treatment = 0
    h_autoencoder_treatment_m = 0
    h_autoencoder_MSE = 0
    h_autoencoder_U = 0
    h_autoencoder_P = 0
    h_autoencoder_P_2 = 0
    h_autoencoder_T_bias = 0
    h_autoencoder_Y_bias = 0
    h_autoencoder_Y_T = 0
    

###set up the architecture of autoencoder  
###the dimension of surrogate confounder is 14  
    Data = np.concatenate((treatment,Y,M),axis=1)

    
    input_U_1 = keras.Input(shape=(3,))
    
    x1 = layers.Dense(3, activation='selu')(input_U_1)
    x1 = layers.Dense(6, activation='selu')(x1)
    x3 = layers.Dense(10, activation='selu', activity_regularizer = UncorrelatedFeaturesConstraint_target(10,treatment.reshape(n,), weightage = 10))(x1)
    
    
    x2 = layers.Dense(4, activation='selu')(x1)
    x2 = layers.Dense(4, activation='selu')(x2)
    treatment_pred = layers.Dense(1, activation='sigmoid')(x2)
    
    encoded = layers.concatenate([x2,x3])
    
    y = layers.Dense(6, activation='selu')(x3)
    y = layers.Dense(3, activation='selu')(y)
    decoded = layers.Dense(3)(y)
    T_M_Y_hat = layers.concatenate([treatment_pred,decoded])
 
    
    autoencoder = keras.Model(inputs = [input_U_1], outputs=[treatment_pred,decoded,T_M_Y_hat])
    encoder = keras.Model(inputs = [input_U_1], outputs= encoded)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02),loss=["binary_crossentropy",outcome_loss_fn,corr_loss_fn(Data = Data),],loss_weights=[lambda_2,1,lambda_1])
 
    while (total_loss_1 - total_loss> 1 and iter<=6) or iter <=2: 

            total_loss_1 = copy.copy(total_loss)  
       
            #### recover residual of M and Y using autoencoder
            M_Y = np.float32(np.concatenate((res_Y,res_M),axis=1))
            X_tr = treatment.reshape(n,1)
            Bias_norm_orth = np.matmul((np.diag(np.ones([n,1]).reshape(n,)) - np.matmul(X_tr,np.matmul(inv(np.matmul(X_tr.transpose(),X_tr)),X_tr.transpose()))),M_Y)
           
            T_M_Y = np.float32(np.concatenate((treatment,Bias_norm_orth),axis=1))
           
            autoencoder.fit(M_Y, [treatment,Bias_norm_orth,T_M_Y],
                    epochs= 2000,   #+ 100*math.floor(iter),
                    batch_size=n,verbose = 0)
      
          
            T_M_Y_hat = autoencoder.predict(M_Y)[2]
           
            encoded_U = encoder.predict(M_Y)
            Uhat = encoded_U
            Uhat = Uhat*(U_scale/np.linalg.norm(Uhat, axis=0))
   
            reg_auto = LinearRegression(fit_intercept=False).fit(Uhat,P)
            np.corrcoef(np.concatenate((reg_auto.predict(Uhat),P),axis=1).transpose())[0,1]   
       
            ##update the loss 
            bce = tf.keras.losses.BinaryCrossentropy()
            
            loss_T = bce(treatment,T_M_Y_hat[:,0].reshape(n,1)).eval(session=sess)      
       
            Loss_Y_M_T = np.concatenate((treatment - T_M_Y_hat[:,0].reshape(n,1),res_Y - T_M_Y_hat[:,1].reshape(n,1),res_M - T_M_Y_hat[:,2:].reshape(n,2)),axis=1)
     
            corr_res = LA.norm(np.corrcoef(Loss_Y_M_T.transpose()) - np.diag(np.ones(4)))**2
       
            total_loss = outcome_loss_fn(M_Y,T_M_Y_hat[:,1:]).eval() + lambda_1*corr_res + lambda_2*loss_T #+ lambda_3*loss_L #+ lambda_4*loss_orth
     
            loss_track.append(total_loss)      
            outcome_track.append(outcome_loss_fn(M_Y,T_M_Y_hat[:,1:]).eval())
        
## update the linear part for modeling M and Y 
    
            k1 = 14
        
            U_est = copy.copy(Uhat)
            X_M_U = np.concatenate((U_est,X[:,0:5],treatment),axis=1)
            reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U, M)
            gamma_M = reg_M_U.coef_[:,0:k1].transpose()
            beta_M = reg_M_U.coef_[:,k1:(k1+5)]
            T_M =  reg_M_U.coef_[:,-1]
            M_int = reg_M_U.intercept_
      
            res_M = M - (M_int + np.matmul(X[:,0:5],beta_M.transpose()).reshape(n,2) + T_M*treatment)
            linear_M = (M_int + np.matmul(X[:,0:5],beta_M.transpose()).reshape(n,2) + T_M*treatment)

            X_Y_U = np.concatenate((treatment,T_M_Y_hat[:,1].reshape(n,1), X[:,5:], M),axis=1)
            reg_Y_U = LinearRegression(fit_intercept=False).fit(X_Y_U, Y)
            beta_Y = reg_Y_U.coef_[0,2:(4)].reshape(2,1)
            beta_Y_M = reg_Y_U.coef_[0,(4):].reshape(2,1)
            #Y_int = reg_Y_U.intercept_
            gamma_Y_bias = reg_Y_U.coef_[0,1]
            beta_Y_T = reg_Y_U.coef_[0,0]
            
            res_Y = Y - (Y_int + np.matmul(X[:,5:],beta_Y).reshape(n,1) + np.matmul(M,beta_Y_M).reshape(n,1) + treatment*beta_Y_T)
            linear_Y = (Y_int + np.matmul(X[:,5:],beta_Y).reshape(n,1) + np.matmul(M,beta_Y_M).reshape(n,1) + treatment*beta_Y_T)
          
           
            T_M_track.append(np.sum(beta_Y_M.reshape(1,2)*T_M))
            Y_T_track.append(beta_Y_T)
           
            if total_loss < loss_ind:
           
                h_autoencoder_T_bias = np.mean(abs(T_M - np.ones([1,2])))     
                h_autoencoder_Y_bias = np.mean(abs(beta_Y_M - 0.5*np.ones([2,1]))) 
                h_autoencoder_treatment_m = np.sum(beta_Y_M.reshape(1,2)*T_M)
                #h_PCA_MSE = LA.norm(np.concatenate((Loss_Y - noise_Y, Loss_M - noise_M),axis=1))/n**0.5
    
                U_his = Uhat
                treat_model = LogisticRegression(solver='liblinear',fit_intercept=False).fit(U_his, treatment.reshape(n,))
                prop_score = treat_model.predict_proba(U_his)
                h_autoencoder_P = np.corrcoef(np.concatenate((prop_score[:,1].reshape(n,1),P),axis=1).transpose())[0,1]
                
                reg_Uhat = LinearRegression(fit_intercept=False).fit(U_his,P)
                h_autoencoder_P_2 =  np.corrcoef(np.concatenate((reg_Uhat.predict(U_his),P),axis=1).transpose())[0,1]
    
    
              
                X_M = np.concatenate((U_his,X[:,0:5],treatment),axis=1)
                reg_M = LinearRegression().fit(X_M, M)
                
                X_Y = np.concatenate((treatment,U_his,X[:,5:],M),axis=1)
                reg_Y = LinearRegression().fit(X_Y, Y)
            
                   
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
               
    
            
            
            iter = iter + 1
            print(iter)
            
    
            
    print(ii)  

## predict the residual in Y based on random forest    
    rf_train = np.concatenate((treatment,M,X),axis=1)
    rf_label = Y_bias
    
    model_y = clone(first_stage_reg().fit(rf_train, rf_label).best_estimator_)
    model_y.fit(rf_train,rf_label.reshape(n,))
    Y_bias_pred = model_y.predict(X = np.concatenate((test_treatment,test_M,test_X),axis=1)).reshape(n1,1)
    pred_Y = np.matmul(test_X[:,5:],beta_Y_auto) + np.matmul(test_M,beta_Y_M_auto) + test_treatment*beta_Y_T_auto + gamma_Y_auto*Y_bias_pred.reshape(n1,1)
    mse_auto[j,ii] = (sum((pred_Y.reshape(n1,1) - test_Y.reshape(n1,1))**2)/n1)**0.5 
 
    
    
      
    autoencoder_T_bias[j,ii] = h_autoencoder_T_bias  
    autoencoder_Y_bias[j,ii] = h_autoencoder_Y_bias
    autoencoder_treatment[j,ii] = h_autoencoder_treatment
 
    autoencoder_iter[j,ii] = iter 
    autoencoder_treatment_m[j,ii] = h_autoencoder_treatment_m
    autoencoder_treatment_d[j,ii] = h_autoencoder_Y_T
    
    
    ################### Causal Forest method
    
 
   
    covariate = np.concatenate((X,M),axis=1)
   
    # #covariate = M
    model_y = clone(first_stage_reg_1().fit(covariate, Y).best_estimator_)
   
    model_t = clone(first_stage_clf().fit(covariate, treatment).best_estimator_)
   
    est = CausalForestDML(model_y=model_y,
                      model_t=model_t,
                      discrete_treatment=True,
                      cv=3,
                      n_estimators=20,
                      random_state=123)
    est.fit(Y, treatment, X=covariate)
    pred = est.effect(X = covariate)
   
    forest_treatment[j,ii] = np.mean(abs(pred - 2))
   
    pred_covariate = np.concatenate((test_X,test_M),axis=1)  
    mse_forest[j,ii] = (sum((est.models_y[0][0].predict(X = pred_covariate).reshape(n1,1) - test_Y.reshape(n1,1))**2)/n1)**0.5 
  
 
    
    #  ############### Xleaner method
    
    X_learner_models = GradientBoostingRegressor(n_estimators=50, max_depth=4, min_samples_leaf=int(n/100))
    X_learner_propensity_model = RandomForestClassifier(n_estimators=100, max_depth=6, 
                                                min_samples_leaf=int(n/100))
    X_learner = XLearner(models=X_learner_models, propensity_model=X_learner_propensity_model)
    # Train X_learner
    X_learner.fit(Y, treatment, X=covariate)
    # Estimate treatment effects on test data
    X_te = X_learner.effect(covariate)
    X_treatment[j,ii] = np.mean(abs(X_te - 2))
    X_learner_models.fit(covariate,Y)
    mse_X[j,ii] = (sum((X_learner_models.predict(pred_covariate).reshape(n1,1) - test_Y.reshape(n1,1))**2)/n1)**0.5 
    
    
    
    
    
    #####LSEM method
    
    X_M_U = np.concatenate((X[:,0:5],treatment),axis=1)
    reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U, M)
    #gamma_M = reg_M_U.coef_[:,0:k].transpose()
    beta_M = reg_M_U.coef_[:,0:5]
    T_M =  reg_M_U.coef_[:,-1]
    M_int = reg_M_U.intercept_
   

    X_Y_U = np.concatenate((treatment,M,X[:,5:]),axis=1)
    reg_Y_U = LinearRegression(fit_intercept=False).fit(X_Y_U, Y)
    beta_Y_M = reg_Y_U.coef_[0,1:3].reshape(2,1)
    beta_Y_T = reg_Y_U.coef_[0,0]
    Y_int = reg_Y_U.intercept_
    beta_Y = reg_Y_U.coef_[0,3:].reshape(2,1)
    
    
    
    LR_T_bias[j,ii] = np.mean(abs(T_M - np.ones([1,2])))     
    LR_Y_bias[j,ii] = np.mean(abs(beta_Y_M - 0.5*np.ones([2,1]))) 

    
    LR_treatment_m[j,ii] = np.sum(beta_Y_M.transpose()*T_M)
    LR_treatment_d[j,ii] = beta_Y_T
    LR_treatment[j,ii] = np.sum(beta_Y_M.transpose()*T_M) + beta_Y_T
    pred_Y_SEM = np.matmul(test_X[:,5:],beta_Y) + np.matmul(test_M,beta_Y_M) + test_treatment*beta_Y_T
    mse_lr[j,ii] = (sum((pred_Y_SEM.reshape(n1,1) - test_Y.reshape(n1,1))**2)/n1)**0.5 
   

###output the results in Table 1 under k = 2
## the true treatment effect is 2, true mediation effect is 1 

np.mean(abs(PCA_treatment-2),1)
np.std(abs(PCA_treatment-2),1)
np.mean(abs(PCA_treatment_m-1),1)
np.std(abs(PCA_treatment_m-1),1)
np.mean(abs(PCA_treatment_d-1),1)
np.std(abs(PCA_treatment_d-1),1)
np.mean(abs(mse_pca),1)
np.std(abs(mse_pca),1)

np.mean(abs(autoencoder_treatment-2),1)
np.std(abs(autoencoder_treatment-2),1)
np.mean(abs(autoencoder_treatment_m-1),1)
np.std(abs(autoencoder_treatment_m-1),1)
np.mean(abs(autoencoder_treatment_d-1),1)
np.std(abs(autoencoder_treatment_d-1),1)
np.mean(abs(mse_auto),1)
np.std(abs(mse_auto),1)

np.mean(abs(LR_treatment-2),1)
np.std(abs(LR_treatment-2),1)
np.mean(abs(LR_treatment_m-1),1)
np.std(abs(LR_treatment_m-1),1)
np.mean(abs(LR_treatment_d-1),1)
np.std(abs(LR_treatment_d-1),1)
np.mean(abs(mse_lr),1)
np.std(abs(mse_lr),1)

np.mean(forest_treatment,1)
np.std(forest_treatment,1)
np.mean(mse_forest,1)
np.std(mse_forest,1)

np.mean(X_treatment,1)
np.std(X_treatment,1)
np.mean(mse_X,1)
np.std(mse_X,1)

           
   
####HIMA method
############# Import packages

import rpy2
print(rpy2.__version__)

from rpy2.rinterface import R_VERSION_BUILD
print(R_VERSION_BUILD)

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1) 


packnames = ('ncvreg', 'doParallel', 'HIMA')
from rpy2.robjects.vectors import StrVector

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
    
    
from rpy2.robjects import FloatVector
from rpy2.robjects import DataFrame
import rpy2.robjects.numpy2ri
from rpy2.robjects import r
import rpy2.robjects as ro
    
rpy2.robjects.numpy2ri.activate()
importr('HIMA')   


###setup and data generation
repeat = 30      
size_num = np.array([200,800,2000]) 

hima_treatment = np.zeros([3,repeat])
hima_treatment_m = np.zeros([3,repeat])
hima_treatment_d =np.zeros([3,repeat])

hima_MSE = np.zeros([3,repeat])
hima_T_bias = np.zeros([3,repeat])
hima_Y_bias = np.zeros([3,repeat])
    

for ii in range(repeat):
    
  np.random.seed(ii)  
  for j in range(3):
      
    N = size_num[j]
        #np.random.seed(123)\
    #N = 2000       
    U = np.zeros([N,1])
    U_ind = np.zeros([N,1])
    for i in range(N):
        ind =  np.random.binomial(1,0.5)
        U_ind[i] = ind
        if ind == 1:
           U[i] = np.random.normal(2,1.5,1)
        else: 
           U[i] = np.random.normal(-2,1.5,1)
           
    P_whole = np.exp(0.4*U)/(1+np.exp(0.4*U))
    
    
    treatment_whole = np.random.binomial(1,P_whole)
    np.corrcoef(np.concatenate((treatment_whole,P_whole),axis=1).transpose()) 
    np.corrcoef(np.concatenate((U,treatment_whole),axis=1).transpose()) 
   
    Bias=  [0.6,0.8,2]*U
   
    Bias = (Bias - np.mean(Bias,axis=0))/np.std(Bias,axis = 0)
    
    
    X_whole = np.random.multivariate_normal(np.zeros(7), np.diag(np.ones(7)), N)
    
    noise_M = np.random.multivariate_normal(np.zeros(2), 0.5*np.diag(np.ones(2)), N)
    coef_M = rvs(dim=5)
    M_whole = 0.5*Bias[:,0:2] + [1,1]*treatment_whole + 0.5*np.matmul(X_whole[:,0:5],coef_M[:,0:2]) + 1*noise_M
    true_M_whole = 0.5*Bias[:,0:2] + [1,1]*treatment_whole + 0.5*np.matmul(X_whole[:,0:5],coef_M[:,0:2])

       
    X_M_t = np.concatenate((treatment_whole, M_whole,),axis=1)
    reg = LinearRegression(fit_intercept=False).fit(X_M_t, Bias[:,2].reshape(N,1))
    np.corrcoef(np.concatenate((reg.predict(X_M_t),Bias[:,2].reshape(N,1)),axis=1).transpose())
    
    noise_Y = np.random.multivariate_normal([0], [[0.3]], N)
    Y_whole = 1*treatment_whole + (1*Bias[:,2]).reshape(N,1) + np.matmul(M_whole,[0.5,0.5]).reshape(N,1) + 0.3*np.matmul(X_whole[:,5:],[1,1]).reshape(N,1) + 1*noise_Y
    true_Y_whole = 1*treatment_whole + 1*Bias[:,2].reshape(N,1) + np.matmul(M_whole,[0.5,0.5]).reshape(N,1) + 0.3*np.matmul(X_whole[:,5:],[1,1]).reshape(N,1)
   
    np.corrcoef(np.concatenate((true_Y_whole,Bias[:,2].reshape(N,1)),axis=1).transpose())
    
    
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
    
    P = P_whole[train_ID]
    test_P = P_whole[test_ID]
    
    n = len(train_ID)
    n1 = len(test_ID)
    
   
########estimate causal effects and mediation effects via HIMA

    treatment_R = robjects.FloatVector(treatment)
    Y_R = robjects.FloatVector(Y)
    nr,nc = M.shape
    M_R = rpy2.robjects.r.matrix(M, nrow=nr, ncol=nc)
    rpy2.robjects.r.assign("M", M_R)
   
    
    HIMA_R = rpy2.robjects.r['hima'](X = treatment_R, Y = Y_R, M = M_R)
    hima_para = np.asarray(HIMA_R)
    num = hima_para.shape[1]

    hima_treatment_m[j,ii] = np.sum(hima_para[3,:])
    hima_treatment[j,ii] = hima_para[2,0]
    hima_treatment_d[j,ii] = hima_para[2,0] - np.sum(hima_para[3,:])
   
    hima_T_bias[j,ii] = np.mean(abs(hima_para[0,:] - np.ones([1,num])))     
    hima_Y_bias[j,ii] = np.mean(abs(hima_para[1,:] - 0.5*np.ones([num,1]))) 
    
    pred_Y_hima = np.matmul(test_M,hima_para[1,:].reshape(2,1)) + test_treatment*(hima_para[2,0] - np.sum(hima_para[3,:]))
    hima_MSE[j,ii] = (sum((pred_Y_hima.reshape(n1,1) - test_Y.reshape(n1,1))**2)/n1)**0.5 



np.mean(abs(hima_treatment-2),1)
np.std(abs(hima_treatment-2),1)
np.mean(abs(hima_treatment_m-1),1)
np.std(abs(hima_treatment_m-1),1)
np.mean(abs(hima_MSE),1)
np.std(abs(hima_MSE),1)