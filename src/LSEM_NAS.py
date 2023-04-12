
import numpy as np
from sklearn.linear_model import LinearRegression

def LSEM_NAS(train_X,train_med,train_outcome,train_treatment,test_X,test_med,test_outcome,test_treatment,n,n_1,m):


    X_M_U = np.concatenate((train_X,train_treatment),axis=1)
    reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U, train_med)
    beta_M = reg_M_U.coef_[:,0:9]
    T_M =  reg_M_U.coef_[:,-1]
    M_int = reg_M_U.intercept_
   
    res_M = train_med - reg_M_U.predict(X_M_U) 
    X_Y_U = np.concatenate((train_treatment,train_med,train_X),axis=1)
    reg_Y_U = LinearRegression(fit_intercept=False).fit(X_Y_U, train_outcome)
    beta_Y_M = reg_Y_U.coef_[0,1:(m+1)].reshape(m,1)
    beta_Y_T = reg_Y_U.coef_[0,0]
    beta_Y = reg_Y_U.coef_[0,(m+1):].reshape(9,1)
    Y_ini =  reg_Y_U.intercept_
      
    treat_m_lr = np.sum(beta_Y_M.transpose()*T_M)
    
    treat_d_lr = beta_Y_T
    
    treat_lr = np.sum(beta_Y_M.transpose()*T_M) + beta_Y_T
    
    pred_Y_SEM =  reg_Y_U.intercept_ + np.matmul(test_X,beta_Y) + np.matmul(test_med,beta_Y_M) + test_treatment*beta_Y_T + Y_ini
 
#    mse_lr[ii] = (sum((pred_Y_SEM.reshape(n_1,1) - test_outcome.reshape(n_1,1))**2)/n_1)**0.5 
    med_lr = np.median(abs(pred_Y_SEM.reshape(n_1,1) - test_outcome.reshape(n_1,1)))
    
    return treat_lr,treat_m_lr,treat_d_lr,med_lr


