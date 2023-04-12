
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import numpy as np

def X_learner_NAS(train_X,train_med,train_outcome,train_treatment,test_X,test_med,test_outcome,n,n_1):
    
    covariate = np.concatenate((train_X,train_med),axis=1)
    X_learner_models = GradientBoostingRegressor(n_estimators=20, max_depth=4, min_samples_leaf=int(n/10))
    X_learner_propensity_model = RandomForestClassifier(n_estimators=20, max_depth=4, 
                                               min_samples_leaf=int(n/10))
    X_learner = XLearner(models=X_learner_models, propensity_model=X_learner_propensity_model)
    # Train X_learner
    X_learner.fit(train_outcome, train_treatment, X=covariate)
    # Estimate treatment effects on test data
    X_te = X_learner.effect(covariate)
    treat_X = np.mean(X_te)
    X_learner_models.fit(covariate,train_outcome)
    
    pred_covariate = np.concatenate((test_X, test_med),axis=1)
    #mse_X[ii] = (sum((X_learner_models.predict(pred_covariate).reshape(n_1,1) - test_outcome.reshape(n_1,1))**2)/n_1)**0.5 
    med_X = np.median(abs(X_learner_models.predict(pred_covariate).reshape(n_1,1) - test_outcome.reshape(n_1,1)))
        
    return treat_X, med_X


