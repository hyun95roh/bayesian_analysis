import pandas as pd 
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import (train_test_split,GridSearchCV, StratifiedKFold, KFold )
from sklearn.tree import (_tree
                          ,plot_tree,
                          DecisionTreeClassifier as DTC)
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier, AdaBoostClassifier
from sklearn.impute import SimpleImputer 
import seaborn as sb 
from sklearn.linear_model import (LinearRegression ,RidgeCV ,LassoCV ) 
from sklearn.decomposition import PCA 
from sklearn.pipeline import make_pipeline 
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,mean_squared_error, mean_absolute_error,make_scorer
from sklearn.feature_selection import RFECV  

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier 
from ucimlrepo import fetch_ucirepo 

############################################
# Other models : RF/XGB/SVC
# fetch dataset 
lung_cancer = fetch_ucirepo(id=62) 
  
# data (as pandas dataframes) 
X = lung_cancer.data.features 
y = lung_cancer.data.targets 
df = pd.concat([X,y],axis=1).dropna() 
X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# Convert target variable to integers starting from 0
label_encoder = LabelEncoder()  
y = label_encoder.fit_transform(y)  

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# XGBoost Classifier
xgb_clf = XGBClassifier(n_estimators=100, random_state=42)
xgb_clf.fit(X_train, y_train)
xgb_pred = xgb_clf.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)

# Support Vector Classifier (SVC)
svc_clf = SVC(kernel='linear')
svc_clf.fit(X_train, y_train)
svc_pred = svc_clf.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_pred)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV for RF
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5],
    'min_samples_split': [2, 5, 10]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=cv, scoring='accuracy')  
grid_search_rf.fit(X_train, y_train)  
best_rf = grid_search_rf.best_estimator_

## Performances 
grid_search_rf.fit(X_train, y_train)
best_param_rf = grid_search_rf.best_params_ 
Train_acc_rf = accuracy_score(y_train,best_rf.predict(X_train))
Test_acc_rf = accuracy_score(y_test,best_rf.predict(X_test))



# GridSearchCV for XGB
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.8, 0.9, 1.0]
}

grid_search_xgb = GridSearchCV(XGBClassifier(random_state=42), param_grid_xgb, cv=cv, scoring='accuracy')
grid_search_xgb.fit(X_train, y_train)
best_xgb = grid_search_xgb.best_estimator_ 
## performance
grid_search_xgb.fit(X_train, y_train)
best_param_xgb = grid_search_xgb.best_params_ 
Train_acc_xgb = accuracy_score(y_train, best_xgb.predict(X_train))
Test_acc_xgb = accuracy_score(y_test, best_xgb.predict(X_test)) 

# GridSearchCV for svc
param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid_search_svc = GridSearchCV(SVC(random_state=42), param_grid_svc, cv=cv, scoring='accuracy')
grid_search_svc.fit(X_train, y_train)
best_svc = grid_search_svc.best_estimator_
## performance
grid_search_svc.fit(X_train, y_train)
best_param_svc = grid_search_svc.best_params_  
Train_acc_svc = accuracy_score(y_train, best_svc.predict(X_train)) 
Test_acc_svc = accuracy_score(y_test, best_svc.predict(X_test)) 


