
import tensorflow as tf
import tensorflow_probability as tfp
from keras import backend as K
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
import copy
from numpy import linalg as LA
from numpy.linalg import inv
from sklearn.linear_model import LogisticRegression
import keras
from keras import layers
import math
from keras.constraints import UnitNorm, Constraint  
sess = tf.InteractiveSession()  

class UncorrelatedFeaturesConstraint_target(Constraint):
    
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
    

def corr_grad_T(Loss_M,treatment,P_est,gamma_U,gamma_t):
    n = Loss_M.shape[0]
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

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

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

def Prop_AE_NAS(train_X,train_outcome,train_med,train_treatment,test_X,test_outcome,test_med,test_treatment,T_M_ini, beta_Y_M_ini,
                beta_M_ini,beta_Y_ini,beta_Y_T_ini,gamma_Y_ini,gamma_Y_bias_ini,x_y_fit_ini,m_y_fit_ini,Y_int_ini,M_int_ini,U_begin,gamma_T,n,n_1,m):

##set up penality parameters

    lambda_1 = 1
    lambda_2 = 0.1
    
    Y_int = Y_int_ini
    res_Y = train_outcome - (Y_int_ini + x_y_fit_ini*beta_Y_ini + m_y_fit_ini*beta_Y_M_ini + train_treatment*beta_Y_T_ini)
    res_Y = res_Y - np.mean(res_Y)
   
    M_int = M_int_ini
    res_M = train_med - (M_int + np.matmul(train_X,beta_M_ini.transpose()).reshape(n,m) + T_M_ini*train_treatment)
    res_M = res_M - np.mean(res_M,axis = 0)
  
    Loss_M_T = np.concatenate((train_treatment,res_Y,res_M),axis=1)
   
    corr_res = LA.norm(np.corrcoef(Loss_M_T.transpose()) - np.diag(np.ones(m+2)))**2
   
    P_est = sigmoid(np.matmul(U_begin,gamma_T))
  
    loss_T = 1000
    
    Loss_total = np.sum(res_Y**2) + np.sum(res_M**2) + lambda_1*corr_res + lambda_2*loss_T #+ lambda_3*loss_L #+ lambda_4*loss_orth
 
    step_size = 0.05
    
    total_loss_1 = copy.copy(Loss_total)
    #loss_L_1 = copy.copy(loss_L)
    #loss_L = 0
    total_loss = 0
    iter = 1
    loss_track = [total_loss_1]
    loss_track_Y = [np.sum(res_Y**2)+np.sum(res_M**2)]
    #loss_track_M = [np.sum(Loss_M**2)]
    loss_track_corr_res = [corr_res**2]
    loss_track_T = [loss_T]
    loss_norm = [0]
   
   
    loss_ind = 100*total_loss_1
    corr_track= []
    
    mse_auto = 0
    MSE_auto = 0
    pred_track_auto = []
    pred_track_auto_mse = []
    
    r_pred_track_auto = []
    r_pred_track_auto_mse = []
    
    Med_effect_auto_1 = 0
    Med_effect_auto_2 = 0
    med_effect_auto_1 = 0
    med_effect_auto_2 = 0
    dir_effect_auto = 0
  
    
    Data = np.float32(np.concatenate((train_treatment,train_outcome,train_med),axis=1))
     
###set up the architecture of autoencoder      
    input_U_1 = keras.Input(shape=(m+1,))
    
    x1 = layers.Dense((m+1), activation='selu')(input_U_1)
    x1 = layers.Dense(int(np.ceil(2*(m+1))), activation='selu')(x1)
    x3 = layers.Dense(int(np.ceil(3*(m+1))), activation='selu', activity_regularizer = UncorrelatedFeaturesConstraint_target(int(np.ceil(3*(m+1))),train_treatment.reshape(n,), weightage = 10))(x1)
 
    x2 = layers.Dense(4, activation='selu')(x1)
    x2 = layers.Dense(4, activation='selu')(x2)
     
    treatment_pred = layers.Dense(1, activation='sigmoid')(x2)
    
    encoded = layers.concatenate([x2,x3])
    
    y = layers.Dense(int(np.ceil(2*(m+1))), activation='selu')(x3)
    y = layers.Dense(int(np.ceil(1*(m+1))), activation='selu')(y)
    decoded = layers.Dense((m+1))(y)
    T_M_Y_hat = layers.concatenate([treatment_pred,decoded])
    
    autoencoder = keras.Model(inputs = [input_U_1], outputs=[treatment_pred,decoded,T_M_Y_hat])
    encoder = keras.Model(inputs = [input_U_1], outputs= encoded)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02),loss=["binary_crossentropy",outcome_loss_fn,corr_loss_fn(Data = Data),],loss_weights=[lambda_2,1,lambda_1])
 
    x_y_fit = x_y_fit_ini
    m_y_fit = m_y_fit_ini
    beta_Y = beta_Y_ini
    beta_Y_T = beta_Y_T_ini
    gamma_Y = gamma_Y_ini
    beta_Y_M = beta_Y_M_ini

    while iter<=5:
       
         total_loss_1 = copy.copy(total_loss)
         M_Y = np.float32(np.concatenate((res_Y,res_M),axis=1))
           
         X_tr = np.concatenate((train_treatment,np.ones([n,1])),axis = 1)
         Bias_norm_orth = np.matmul((np.diag(np.ones([n,1]).reshape(n,)) - np.matmul(X_tr,np.matmul(inv(np.matmul(X_tr.transpose(),X_tr)),X_tr.transpose()))),M_Y)
         T_M_Y = np.float32(np.concatenate((train_treatment,res_Y,res_M),axis=1))
           
         autoencoder.fit(M_Y, [train_treatment,Bias_norm_orth,T_M_Y],
                    epochs= 1500,   #+ 100*math.floor(iter),
                    batch_size= n,verbose = 1)
         
     
         T_M_Y_hat = autoencoder.predict(M_Y)[2]
         if math.isnan(np.sum(T_M_Y_hat)):
               break
         
         encoded_U = encoder.predict(M_Y)
         Uhat = encoded_U
          
         bce = tf.keras.losses.BinaryCrossentropy()
            
         loss_T = bce(train_treatment,T_M_Y_hat[:,0].reshape(n,1)).eval(session=sess)      
       
         Loss_Y_M_T = np.concatenate((train_treatment - T_M_Y_hat[:,0].reshape(n,1),res_Y - T_M_Y_hat[:,1].reshape(n,1),res_M - T_M_Y_hat[:,2:].reshape(n,m)),axis=1)
     
         corr_res = LA.norm(np.corrcoef(Loss_Y_M_T.transpose()) - np.diag(np.ones(m+2)))**2
       
         total_loss = outcome_loss_fn(M_Y,T_M_Y_hat[:,1:]).eval() + lambda_1*corr_res + lambda_2*loss_T
     
         loss_track.append(total_loss) 
         
         k1 = int(np.ceil(3*(m+1)))+4
        
         U_est = copy.copy(Uhat)
         X_M_U = np.concatenate((U_est,train_X,train_treatment),axis=1)
         reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U, train_med)
         gamma_M = reg_M_U.coef_[:,0:k1].transpose()
         beta_M = reg_M_U.coef_[:,k1:(k1+9)]
         T_M =  reg_M_U.coef_[:,-1]
         M_int = reg_M_U.intercept_
      
         res_M = train_med - (M_int + np.matmul(train_X,beta_M.transpose()).reshape(n,m) + T_M*train_treatment)
         res_M = res_M - np.mean(res_M,axis =0)
         
         Loss_M_pred = train_med - (np.matmul(train_X[:,0:9],beta_M.transpose()).reshape(n,m) +  train_treatment*T_M)
         
        ######
        ##update rf mediator
          
         m_y_outcome = train_outcome - x_y_fit*beta_Y - train_treatment*beta_Y_T -  T_M_Y_hat[:,1].reshape(n,1) -Y_int
         rf_m_y = RandomForestRegressor(max_depth=6, min_samples_leaf=5, random_state=123)
         rf_m_y.fit(train_med,m_y_outcome.reshape(n,))
         m_y_fit = rf_m_y.predict(train_med).reshape(n,1)
        
        ##update rf x
        
         x_y_outcome = train_outcome - m_y_fit*beta_Y_M - train_treatment*beta_Y_T -  T_M_Y_hat[:,1].reshape(n,1) -Y_int
         rf_x_y = RandomForestRegressor(max_depth=6, min_samples_leaf=5, random_state=123)
         rf_x_y.fit(train_X,x_y_outcome.reshape(n,))
         x_y_fit = rf_x_y.predict(train_X).reshape(n,1)
       
        
        ##update y
        
         
         Y_bias = T_M_Y_hat[:,1].reshape(n,1)
         X_Y_U = np.concatenate((train_treatment.reshape(n,1),Y_bias,x_y_fit,m_y_fit),axis=1)
         reg_Y_U = LinearRegression(fit_intercept=True).fit(X_Y_U, train_outcome)
         beta_Y = reg_Y_U.coef_[0,2]
         beta_Y_M = reg_Y_U.coef_[0,3]
         Y_int = reg_Y_U.intercept_
         gamma_Y_bias = 1
         beta_Y_T = reg_Y_U.coef_[0,0]
        
        
         M_pred  =  beta_Y_M*rf_m_y.predict(test_med).reshape(n_1,1) + beta_Y*rf_x_y.predict(test_X).reshape(n_1,1)
     
         ### calculate mediator effect 1
         X_M_U_1 = np.concatenate((U_est,train_X,np.ones(n).reshape(n,1)),axis=1)
         X_M_U_0 = np.concatenate((U_est,train_X,np.zeros(n).reshape(n,1)),axis=1)
         
         Med_effect_auto_1 = np.mean(rf_m_y.predict(reg_M_U.predict(X_M_U_1)).reshape(n,1)- rf_m_y.predict(reg_M_U.predict(X_M_U_0)).reshape(n,1))
         
        ### calculate mediator effect 2
         med_1_fak = reg_M_U.predict(X_M_U_1)
         med_0_fak = reg_M_U.predict(X_M_U_0)
        
         med_1 = np.zeros([n,m])
         med_0 = np.zeros([n,m])
         for i in range(n):
             if train_treatment[i] == 1:
                 med_1[i,:] = train_med[i,:]
                 med_0[i,:] = med_0_fak[i,:]
             else:
                 med_1[i,:] = med_1_fak[i,:]
                 med_0[i,:] = train_med[i,:]
        
         Med_effect_auto_2 = np.mean(rf_m_y.predict(med_1).reshape(n,1) - rf_m_y.predict(med_0).reshape(n,1))
   
     
        ######
         res_Y = train_outcome - (Y_int + x_y_fit*beta_Y + m_y_fit*beta_Y_M + train_treatment*beta_Y_T)
         res_Y = res_Y - np.mean(res_Y,axis = 0)
         rf_train = np.concatenate((train_treatment,train_med),axis=1)
         rf_label = Y_bias 
         rf = RandomForestRegressor(n_estimators = 50, max_depth=6,random_state = 123)
 
          
         rf.fit(rf_train,rf_label.reshape(n,))
         Y_bias_pred = rf.predict(np.concatenate((test_treatment,test_med),axis=1)).reshape(n_1,1)
         
         pred_Y = M_pred + test_treatment*beta_Y_T + gamma_Y_bias*Y_bias_pred.reshape(n_1,1) + Y_int
          
         pred_track_auto_mse.append((sum((pred_Y.reshape(n_1,1) - test_outcome.reshape(n_1,1))**2)/n_1)**0.5)
         pred_track_auto.append(np.median(abs(pred_Y.reshape(n_1,1) - test_outcome.reshape(n_1,1))))
         
      
         
         if min(pred_track_auto) < loss_ind:
             
    
                U_his = Uhat
                  
                med_effect_auto_1 = Med_effect_auto_1
                med_effect_auto_2 = Med_effect_auto_2
                dir_effect_auto = beta_Y_T
                
                Y_bias = T_M_Y_hat[:,1].reshape(n,1)
                 
                
                loss_ind  = min(pred_track_auto)
                Med_auto = min(pred_track_auto)
                Mse_auto = min(pred_track_auto_mse)
                
                beta_Y_M_auto = beta_Y_M
    
         iter = iter + 1
         #print(iter)
          
    treat_m_auto_1 = med_effect_auto_1*beta_Y_M_auto
    treat_m_auto_2 = med_effect_auto_2*beta_Y_M_auto
    treat_d_auto = dir_effect_auto
    med_auto_3 = Med_auto
    #mse_auto_3[ii] = Mse_auto
    treat_auto = dir_effect_auto + med_effect_auto_1*beta_Y_M_auto
    
    return treat_m_auto_1, treat_m_auto_2,treat_d_auto, med_auto_3, treat_auto   
    
    
    
    
    
    
    
    