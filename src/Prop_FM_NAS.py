



import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
import copy
from numpy import linalg as LA
from numpy.linalg import inv
from sklearn.linear_model import LogisticRegression

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

def Prop_FM_NAS(train_X,train_outcome,train_med,train_treatment,test_X,test_outcome,test_med,test_treatment,n,n_1,m):

    k = 12
#initialize the outcome and mediators model based on random forest 
       
    rf_m_y = RandomForestRegressor(max_depth=6, min_samples_leaf=5, random_state=123)
    rf_m_y.fit(train_med,train_outcome.reshape(n,))
    m_y_fit = rf_m_y.predict(X = train_med).reshape(n,1)
    m_y_test = rf_m_y.predict(X = test_med).reshape(n_1,1)
    rf_x_y = RandomForestRegressor(max_depth=6, min_samples_leaf=5, random_state=123)
    rf_x_y.fit(train_X,train_outcome.reshape(n,))
    x_y_fit = rf_x_y.predict(X = train_X).reshape(n,1)
    x_y_test = rf_x_y.predict(X = test_X).reshape(n_1,1)
    
    ####
    
    X_Y_U = np.concatenate((m_y_fit.reshape(n,1),x_y_fit.reshape(n,1),train_treatment),axis=1)
    reg_Y_U = LinearRegression(fit_intercept=True).fit(X_Y_U, train_outcome)
    res_Y = train_outcome - reg_Y_U.predict(X_Y_U)
    
    X_M_U = np.concatenate((train_X,train_treatment),axis=1) 
    reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U,train_med)
    res_M = train_med - reg_M_U.predict(X_M_U)

##obtain the initial of U  
     
    U_scale = 1
    pca = PCA(n_components=k)
    pca.fit(np.concatenate((res_Y.reshape(n,1),res_M),axis=1))
    pca.components_.T
    pca.explained_variance_ratio_
    #np.corrcoef(np.concatenate((pca.fit_transform(M - reg_M.predict(X_M_r)),U),axis=1).transpose())    
    U_ini = pca.fit_transform(np.concatenate((res_Y.reshape(n,1),res_M),axis=1)) #+ 2*np.random.randn(n, k) 
    U_ini = ((U_scale/np.linalg.norm(U_ini, axis=0))*U_ini).reshape(n,k)
    #Uhat = np.zeros([n,k])
    Uhat = copy.copy(U_ini)
    U_begin = copy.copy(U_ini)

    lambda_1 = 10
    lambda_2 = 0.1
    
    U_est = copy.copy(U_ini)

##initialize the parameters        
    X_M_U = np.concatenate((U_est,train_X,train_treatment.reshape(n,1)),axis=1)
    reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U, train_med)
    gamma_M = reg_M_U.coef_[:,0:k].transpose()
    beta_M = reg_M_U.coef_[:,k:(k+9)]
    T_M =  reg_M_U.coef_[:,-1]
    M_int = reg_M_U.intercept_

   
    X_Y_U = np.concatenate((train_treatment.reshape(n,1),U_est,x_y_fit,m_y_fit),axis=1)
    reg_Y_U = LinearRegression(fit_intercept=True).fit(X_Y_U, train_outcome)
    beta_Y = reg_Y_U.coef_[0,k+1]
    beta_Y_M = reg_Y_U.coef_[0,k+2]
    Y_int = reg_Y_U.intercept_
    gamma_Y = reg_Y_U.coef_[0,1:(k+1)].reshape(k,1)
    beta_Y_T = reg_Y_U.coef_[0,0]

    
    Loss_Y = train_outcome - (Y_int + x_y_fit*beta_Y + m_y_fit*beta_Y_M + train_treatment*beta_Y_T)
    #Loss_Y = train_outcome - reg_Y_U.predict(X_Y_U)
    Loss_Y_center_U =  Loss_Y - np.mean(Loss_Y,axis = 0)   
    #reg_Y_bias = LinearRegression(fit_intercept=False).fit(U_begin, Loss_Y_center_U)
    reg_Y_bias = LinearRegression(fit_intercept=True).fit(U_begin, Loss_Y)
    gamma_Y = reg_Y_bias.coef_.transpose()

    
    Loss_M = train_med - reg_M_U.predict(X_M_U)
    Loss_M_center_U =  Loss_M - np.mean(Loss_M,axis = 0)  

    gamma_T = 0.1*np.ones(k).reshape(k,1)
    T_int = 0
    P_est = sigmoid(np.matmul(U_begin,gamma_T)+T_int)
    
    Loss_M_T = np.concatenate((train_treatment - P_est,Loss_Y_center_U,Loss_M_center_U),axis=1)
    
    corr_res = LA.norm(np.corrcoef(Loss_M_T.transpose()) - np.diag(np.ones(m+2)))**2
    loss_T = -np.sum(np.log(P_est)*train_treatment + np.log(1-P_est)*(1-train_treatment)) #- np.sum(P_est*np.log(P_est))
  
    scale = 1/n
    
    Loss_total = np.sum(Loss_Y_center_U**2) + np.sum(Loss_M_center_U**2) + lambda_1*corr_res + lambda_2*loss_T #+ lambda_3*loss_L #+ lambda_4*loss_orth
    
    outcome_loss_track = [np.sum(Loss_Y_center_U**2) + np.sum(Loss_M_center_U**2)]
    
     
    
    step_size = 0.00002
    
    total_loss_1 = copy.copy(Loss_total)
    
    total_loss = 0
    iter = 1
    loss_track = [total_loss_1]
    loss_track_1 = [np.sum(Loss_Y_center_U**2) + np.sum(Loss_M_center_U**2)]
    loss_track_Y = [np.sum(Loss_Y**2)]
    loss_track_M = [np.sum(Loss_M**2)]
    loss_track_corr_res = [corr_res]
    loss_track_T = [loss_T]
    
    loss_ind = 1000*total_loss_1
    
    
    #T_M_ini = 0 
    beta_Y_M_ini = 0
    #beta_M_ini = 0
    beta_Y_ini = 0
    beta_Y_T_ini = 0
    
    h_PCA_treatment = 0
    h_PCA_treatment_m = 0
    h_PCA_MSE = 0
    h_PCA_Y_T = 0
    med_effect_1 = 0
    med_effect_2 = 0
    M_pred_test = np.zeros([n_1,1])
    
    x_y_fit_ini =  np.zeros([n,1])
    m_y_fit_ini =  np.zeros([n,1])
    
    med_effect_1_track = []
    med_effect_2_track = []
    dir_effect_track = []
    pred_track = []
    pred_track_mse = []
    r_pred_track = []
    r_pred_track_mse = []
    
    fit_track = []
    
    Mse_fa = 0
    mse_fa_3 = 0
    
    pred_track_res = []
    
    
    while iter<= 100:
##update surrogate confounder 
        
          total_loss_1 = copy.copy(total_loss)
           
          Uhat = np.matmul((np.diag(np.ones([n,1]).reshape(n,)) - np.matmul(train_X,np.matmul(inv(np.matmul(train_X.transpose(),train_X)),train_X.transpose()))),Uhat)
   
          grad_U_k = np.zeros([n,k])
            
          for kk in range(k):
                
                gamma_U = gamma_M[kk,:]
                
                gamma_t = gamma_T[kk,0]
                #grad_corr = corr_grad(Loss_M,gamma_U) 
                
                Loss_M_Y = np.concatenate((Loss_Y_center_U,Loss_M_center_U),axis=1)
                gamma_U_Y = np.concatenate((gamma_Y[kk,:],gamma_U))
                
                grad_corr = corr_grad_T(Loss_M_Y,train_treatment,P_est,gamma_U_Y,gamma_t)   
                
                grad_U_k[:,kk] = ( - 2*gamma_Y[kk,:]*Loss_Y_center_U  - np.sum(2*gamma_M[kk,:]*Loss_M_center_U,1).reshape(n,1) + lambda_1*grad_corr.reshape(n,1) + lambda_2*(-gamma_T[kk,0]*(train_treatment - P_est)).reshape(n,1)).reshape(n,)

          Uhat = Uhat - step_size*grad_U_k
          Uhat = Uhat*(U_scale/np.linalg.norm(Uhat, axis=0))
          
          ##update rf mediator
          
          m_y_outcome = train_outcome - x_y_fit*beta_Y - train_treatment*beta_Y_T - np.matmul(Uhat,gamma_Y).reshape(n,1) - Y_int
          
          rf_m_y.fit(train_med,m_y_outcome.reshape(n,))
          m_y_fit = rf_m_y.predict(train_med).reshape(n,1)
          
          ##update rf x
          
          x_y_outcome = train_outcome - m_y_fit*beta_Y_M - train_treatment*beta_Y_T - np.matmul(Uhat,gamma_Y).reshape(n,1) - Y_int
          rf_x_y.fit(train_X,x_y_outcome.reshape(n,))
          x_y_fit = rf_x_y.predict(train_X).reshape(n,1)
         
          
          ##update y
          
          U_est = copy.copy(Uhat)
          Y_bias = np.matmul(U_est,gamma_Y)
          X_Y_U = np.concatenate((train_treatment.reshape(n,1),Y_bias,x_y_fit,m_y_fit),axis=1)
          reg_Y_U = LinearRegression(fit_intercept=True).fit(X_Y_U, train_outcome)
          beta_Y = reg_Y_U.coef_[0,2]
          beta_Y_M = reg_Y_U.coef_[0,3]
          Y_int = reg_Y_U.intercept_
          gamma_Y_bias = reg_Y_U.coef_[0,1]
          beta_Y_T = reg_Y_U.coef_[0,0]
          
          
          
          M_pred  =  beta_Y_M*rf_m_y.predict(test_med).reshape(n_1,1) + beta_Y*rf_x_y.predict(test_X).reshape(n_1,1)
          #r_M_pred  =  beta_Y_M*rf_m_y.predict(r_med).reshape(n_r,1) + beta_Y*rf_x_y.predict(r_X).reshape(n_r,1)
          
          
          ##update m
         
          X_M_U = np.concatenate((U_est, train_X,train_treatment.reshape(n,1)),axis=1)
          reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U, train_med)
          gamma_M = reg_M_U.coef_[:,0:k].transpose()
          beta_M = reg_M_U.coef_[:,k:(k+9)]
          T_M =  reg_M_U.coef_[:,-1]
          M_int = reg_M_U.intercept_
          
          
          #reg_U = LinearRegression(fit_intercept=False).fit(U_est, train_treatment.reshape(n,))
          #np.corrcoef(np.concatenate((train_treatment.reshape(n,1),reg_U.predict(U_est).reshape(n,1)),axis=1).transpose())          
          #np.corrcoef(np.concatenate((train_treatment.reshape(n,1),train_outcome.reshape(n,1)),axis=1).transpose())          

          ##update P

          treat_model = LogisticRegression(fit_intercept=False).fit(U_est, train_treatment.reshape(n,))
          P_est = treat_model.predict_proba(U_est)[:,1].reshape(n,1)
          gamma_T = treat_model.coef_.transpose()
         
         ### calculate mediator effect 1
          X_M_U_1 = np.concatenate((U_est,train_X,np.ones(n).reshape(n,1)),axis=1)
          X_M_U_0 = np.concatenate((U_est,train_X,np.zeros(n).reshape(n,1)),axis=1)
          
          Med_effect_1 = np.mean(rf_m_y.predict(reg_M_U.predict(X_M_U_1)).reshape(n,1)- rf_m_y.predict(reg_M_U.predict(X_M_U_0)).reshape(n,1))
          
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
         
          Med_effect_2 = np.mean(rf_m_y.predict(med_1).reshape(n,1) - rf_m_y.predict(med_0).reshape(n,1))
        
          ##update loss
            
          Loss_Y = train_outcome - (Y_int + x_y_fit*beta_Y + m_y_fit*beta_Y_M + train_treatment*beta_Y_T)
          #Loss_Y = train_outcome - (np.matmul(train_X,beta_Y).reshape(n,1) + np.matmul(train_med,beta_Y_M).reshape(n,1) + train_treatment*beta_Y_T)
          reg_Y_bias = LinearRegression(fit_intercept=True).fit(U_est, Loss_Y) #- np.mean(Loss_Y))
          #reg_Y_bias = LinearRegression(fit_intercept=False).fit(U_est, Loss_Y)
          gamma_Y = reg_Y_bias.coef_.transpose()
        
          Loss_Y_U = train_outcome - (Y_int + x_y_fit*beta_Y + m_y_fit*beta_Y_M + train_treatment*beta_Y_T + np.matmul(U_est, gamma_Y))
          #Loss_Y_U = train_outcome - (np.matmul(train_X,beta_Y).reshape(n,1) + np.matmul(train_med,beta_Y_M).reshape(n,1) + train_treatment*beta_Y_T + np.matmul(U_est, gamma_Y))
          Loss_Y_center_U =  Loss_Y_U - np.mean(Loss_Y_U,axis = 0) 
          fit_track.append((sum(Loss_Y_U**2)/n)**0.5)

          Loss_M_U = train_med - reg_M_U.predict(X_M_U)
          Loss_M_center_U =  Loss_M_U - np.mean(Loss_M_U,axis = 0)  
          Loss_M_pred = train_med - (np.matmul(train_X[:,0:9],beta_M.transpose()).reshape(n,m) +  train_treatment*T_M)
          
          Loss_M_T = np.concatenate((train_treatment - P_est,Loss_Y_center_U,Loss_M_center_U),axis=1)
  
    
          corr_res = LA.norm(np.corrcoef(Loss_M_T.transpose()) - np.diag(np.ones(m+2)))**2
    
          loss_T = -np.sum(np.log(P_est)*train_treatment + np.log(1-P_est)*(1-train_treatment)) #- np.sum(P_est*np.log(P_est))

          total_loss = np.sum(Loss_Y_center_U**2) + np.sum(Loss_M_center_U**2) + lambda_1*corr_res + lambda_2*loss_T #+ lambda_3*loss_L #+ lambda_4*loss_orth
          
          iter = iter + 1

          loss_track.append(total_loss)
          loss_track_1.append(np.sum(Loss_Y_center_U**2) + np.sum(Loss_M_center_U**2))
          loss_track_corr_res.append(corr_res)
          loss_track_T.append(loss_T)
          
          med_effect_1_track.append(Med_effect_1)
          med_effect_2_track.append(Med_effect_2)
          dir_effect_track.append(beta_Y_T)
          
          #######  out-sample prediction
          
          rf_train = np.concatenate((train_treatment,train_med),axis=1)
          rf_label = Y_bias 

          rf = RandomForestRegressor(n_estimators = 100, max_depth=12,random_state = 123)
          #rf = RandomForestRegressor(n_estimators = 50, max_depth=6,random_state = 123)

   
          Loss_M_U_test = test_med - (np.matmul(test_X[:,0:9],beta_M.transpose()).reshape(n_1,m) +  test_treatment*T_M)
          #r_Loss_M_U_test = r_med - (np.matmul(r_X[:,0:9],beta_M.transpose()).reshape(n_r,m) +  r_treatment*T_M)
        
          rf.fit(rf_train,rf_label.reshape(n,))
          #Y_bias_pred = rf.predict(np.concatenate((test_treatment,Loss_M_U_test),axis=1)).reshape(n_1,1)
          Y_bias_pred = rf.predict(np.concatenate((test_treatment,test_med),axis=1)).reshape(n_1,1)

          #r_Y_bias_pred = rf.predict(np.concatenate((r_treatment,r_Loss_M_U_test),axis=1)).reshape(n_r,1)
    
          pred_Y = M_pred + test_treatment*beta_Y_T + gamma_Y_bias*Y_bias_pred.reshape(n_1,1) + Y_int
          #r_pred_Y = r_M_pred + r_treatment*beta_Y_T + gamma_Y_bias*r_Y_bias_pred.reshape(n_r,1) + Y_int
          
          mse_pca_2 = (sum((pred_Y.reshape(n_1,1) - test_outcome.reshape(n_1,1))**2)/n_1)**0.5
          #mse_pca_2_r = (sum((r_pred_Y.reshape(n_r,1) - r_outcome.reshape(n_r,1))**2)/n_r)**0.5
          pred_track_mse.append(mse_pca_2)
          #r_pred_track_mse.append(mse_pca_2_r)
          
          mse_pca_3 = np.median(abs(pred_Y.reshape(n_1,1) - test_outcome.reshape(n_1,1)))
          #mse_pca_3_r = np.median(abs(r_pred_Y.reshape(n_r,1) - r_outcome.reshape(n_r,1)))
          pred_track.append(mse_pca_3)
          #r_pred_track.append(mse_pca_3_r)
          
          
            
          if total_loss < loss_ind:
                
            
                  
                U_his = Uhat
                X_Y = np.concatenate((train_treatment,U_his,x_y_fit,m_y_fit),axis=1)
                reg_Y = LinearRegression(fit_intercept=False).fit(X_Y, train_outcome)
                Y_bias = np.matmul(U_his,reg_Y.coef_[0,1:(k+1)].transpose()).reshape(n,1)
                
                h_PCA_Y_T = beta_Y_T 
                
                
                loss_ind  = total_loss
                M_pred_test = M_pred 
                
                med_effect_1 = Med_effect_1
                med_effect_2 = Med_effect_2
                
                T_M_ini = T_M
                beta_Y_M_ini = beta_Y_M
                beta_M_ini = beta_M
                beta_Y_ini = beta_Y
                beta_Y_T_ini = beta_Y_T   
                gamma_Y_ini = gamma_Y
                gamma_Y_bias_ini = gamma_Y_bias
                x_y_fit_ini =  x_y_fit
                m_y_fit_ini =  m_y_fit
                Y_int_ini = Y_int
                M_int_ini = M_int
                
                Med_fa = min(pred_track)
                Mse_fa = min(pred_track_mse)
                
               
                
          #print(iter)
          
          
    
    
    med_fa = Med_fa
    treat_m_fa_1 = med_effect_1*beta_Y_M_ini
    treat_m_fa_2 = med_effect_2*beta_Y_M_ini
    treat_d_fa = beta_Y_T
    treat_fa = beta_Y_T + med_effect_1*beta_Y_M_ini
    
    return med_fa, treat_m_fa_1, treat_m_fa_2,treat_d_fa, treat_fa, T_M_ini, beta_Y_M_ini,beta_M_ini,\
           beta_Y_ini, beta_Y_T_ini,gamma_Y_ini,gamma_Y_bias_ini,x_y_fit_ini,m_y_fit_ini,Y_int_ini,M_int_ini,U_begin,gamma_T   
    
    
    
    
    
    
    

