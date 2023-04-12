

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import rpy2.robjects as robjects
import rpy2
import numpy as np

def HIMA_NAS(train_X,train_med,train_outcome,train_treatment,test_X,test_med,test_outcome,test_treatment,n,n_1,m):

    treatment_R = robjects.FloatVector(train_treatment)
    Y_R = robjects.FloatVector(train_outcome)
    nr,nc = train_med.shape
    M_R = rpy2.robjects.r.matrix(train_med, nrow=nr, ncol=nc)
    rpy2.robjects.r.assign("M", M_R)
   
    
    HIMA_R = rpy2.robjects.r['hima'](X = treatment_R, Y = Y_R, M = M_R)
    hima_para = np.asarray(HIMA_R)
    num = hima_para.shape[1]
    select_med = np.int16(np.asarray(HIMA_R.rownames))-1
    
    treat_m_hima = np.sum(hima_para[3,:])
    treat_hima = hima_para[2,0]
    treat_d_hima = hima_para[2,0] - np.sum(hima_para[3,:])
   
     
    pred_Y_hima = np.matmul(test_med[:,select_med],hima_para[1,:].reshape(num,1)) + test_treatment*(hima_para[2,0] - np.sum(hima_para[3,:]))
#    mse_hima[ii] = (sum((pred_Y_hima.reshape(n_1,1) - test_outcome.reshape(n_1,1))**2)/n_1)**0.5 
    med_hima = np.median(abs(pred_Y_hima.reshape(n_1,1) - test_outcome.reshape(n_1,1)))
    
    return treat_hima,treat_m_hima, treat_d_hima, med_hima




