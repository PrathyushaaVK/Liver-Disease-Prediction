# models.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def random_forest_model(X_train, Y_train):
    param_grid = {
        'n_estimators': range(100, 201, 5), 
        'criterion': ['gini', 'entropy'], 
        'max_features': ['auto', 'sqrt', 'log2', None]
    }
    
    tuning = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, verbose=2, scoring='f1')
    tuning.fit(X_train, Y_train)
    print("Best Params for Random Forest:", tuning.best_params_)
    
    return tuning.best_estimator_

def logistic_regression_model(X_train, Y_train):
    paramgrid = [{'max_iter': range(1, 30, 1), 'solver': ['liblinear', 'saga', 'lbfgs']}]
    
    tuningLR = GridSearchCV(estimator=LogisticRegression(), param_grid=paramgrid, cv=5, verbose=True, scoring='f1')
    tuningLR.fit(X_train, Y_train)
    print("Best Params for Logistic Regression:", tuningLR.best_params_)
    
    return tuningLR.best_estimator_
