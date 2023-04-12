#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import rpy2.robjects as robjects
import rpy2
import numpy as np

def HIMA(X,M,Y,treatment,test_X,test_M,test_Y,test_treatment,n1,m):

    treatment_R = robjects.FloatVector(treatment)
    Y_R = robjects.FloatVector(Y)
    nr,nc = M.shape
    M_R = rpy2.robjects.r.matrix(M, nrow=nr, ncol=nc)
    rpy2.robjects.r.assign("M", M_R)
    
    HIMA_R = rpy2.robjects.r['hima'](X = treatment_R, Y = Y_R, M = M_R)
    hima_para = np.asarray(HIMA_R)
    num = hima_para.shape[1]
    
    hima_treatment_m = np.sum(hima_para[3,:])
    hima_treatment = hima_para[2,0]
    hima_treatment_d  = hima_para[2,0] - np.sum(hima_para[3,:])
    
    pred_Y_hima = np.matmul(test_M,hima_para[1,:].reshape(m,1)) + test_treatment*(hima_para[2,0] - np.sum(hima_para[3,:]))
    hima_MSE = (sum((pred_Y_hima.reshape(n1,1) - test_Y.reshape(n1,1))**2)/n1)**0.5 
    
    return hima_treatment, hima_treatment_m, hima_MSE


