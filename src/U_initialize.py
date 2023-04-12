#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.decomposition import PCA
from numpy import linalg as LA
import numpy as np

def U_initialize(M,Y,k,n,U_scale,scale):

    pca = PCA(n_components=k)
    pca.fit(np.concatenate((Y,M),axis=1))
    U_ini = pca.fit_transform(np.concatenate((Y,M),axis=1)) #+ 2*np.random.randn(n, k) 
    if scale == True:
      U_ini = ((U_scale/np.linalg.norm(U_ini, axis=0))*U_ini).reshape(n,k)
      return U_ini
    else:
      return U_ini 
  
    