#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
from sklearn.linear_model import LogisticRegression
import copy

def T_para(Uhat,treatment,n):
    U_est = copy.copy(Uhat)
    treat_model = LogisticRegression(fit_intercept=False).fit(U_est, treatment.reshape(n,))
    P_est = treat_model.predict_proba(U_est)[:,1].reshape(n,1)
    gamma_T = treat_model.coef_.transpose()
    return P_est, gamma_T

