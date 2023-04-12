
import numpy as np
from econml.dml import CausalForestDML
from sklearn.base import clone
from econml.sklearn_extensions.model_selection import GridSearchCVList
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

def first_stage_clf():
    return GridSearchCVList([LogisticRegression()],
                             param_grid_list=[{'C': [0.01, .1, 1, 10, 100]}],
                             cv=5,
                             scoring='neg_mean_squared_error')  


def Causal_Forest_NAS(train_X,train_med,train_outcome,train_treatment,test_X,test_med,test_outcome,n_1):

    covariate = np.concatenate((train_X,train_med),axis=1)

    model_y = clone(first_stage_reg_1().fit(covariate, train_outcome).best_estimator_)

    model_t = clone(first_stage_clf().fit(covariate, train_treatment).best_estimator_)

    est = CausalForestDML(model_y=model_y,
                      model_t=model_t,
                      discrete_treatment=False,
                      cv=3,
                      n_estimators=20,
                      random_state=123)
    
    est.fit(train_outcome, train_treatment, X=covariate)
    
    pred = est.effect(X = covariate)
    
    treat_forest = np.mean((pred))

    pred_covariate = np.concatenate((test_X, test_med),axis=1)  
    
    #mse_forest[ii] = (sum((est.models_y[0][0].predict(X = pred_covariate).reshape(n_1,1) - test_outcome.reshape(n_1,1))**2)/n_1)**0.5 
    med_forest = np.median(abs(est.models_y[0][0].predict(X = pred_covariate).reshape(n_1,1) - test_outcome.reshape(n_1,1)))
    
    return treat_forest, med_forest


 
