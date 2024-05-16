import pandas as pd 
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go 
from matplotlib.pyplot import (subplots, scatter, figure, title, show, xlabel, ylabel, legend, contourf, colorbar, subplots_adjust )
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import (train_test_split,GridSearchCV, StratifiedKFold, KFold )
from sklearn.tree import (_tree
                          ,plot_tree,
                          DecisionTreeClassifier as DTC)
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier, AdaBoostClassifier
from ISLP.bart import BART
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
from mpl_toolkits.mplot3d import Axes3D
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
lung_cancer = fetch_ucirepo(id=62) 
  
# data (as pandas dataframes) 
X = lung_cancer.data.features 
y = lung_cancer.data.targets 
df = pd.concat([X,y],axis=1).dropna() 
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Classification Tree: 
# Decision Tree Classifier (criterion='gini') 
# Grid Search
param_grid= {'ccp_alpha': np.linspace(0, 0.1, 5),
             'max_depth': [None, 3, 5]
             }  
tree_clf = DTC(random_state=0) 
cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=0)  
grid_search = GridSearchCV(tree_clf, param_grid, cv= cv, scoring='accuracy')  
grid_search.fit(X,y) 

best_alpha = grid_search.best_params_['ccp_alpha']  
best_depth = grid_search.best_params_['max_depth'] 
best_score = grid_search.best_score_
print("Best Cost-complexity pruning alpha:", best_alpha) 

# Fit a decision tree clf with ccp 
tree_clf = DTC(ccp_alpha= best_alpha, max_depth= best_depth) 
tree_clf.fit(X, y) 

# Get feature importaances 
importances = tree_clf.feature_importances_ 

# Get indicies of top two features 
top_indicies = np.argsort(importances)[-2:] 
column_names = X.columns 
top_features_name = [ column_names[0:][i] for i in top_indicies]
X_top = X.iloc[:, top_indicies] 


# Plot training points
fig = figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d') 
ax.view_init(elev=90, azim=60)
ax.scatter(X_top.iloc[:, 0], X_top.iloc[:, 1], y, c=y, cmap='viridis', label='Data') 

# Set meshgrid of the decision surface 
x_min, x_max = X_top.iloc[:, 0].min() -1, X_top.iloc[:, 0].max() +1 
y_min, y_max = X_top.iloc[:, 1].min() -1, X_top.iloc[:, 1].max() +1  
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))  

# Using plotly 
# Assuming X_top and y are your data frames
fig = px.scatter_3d(X_top, x=X_top.columns[0], y=X_top.columns[1], z=y, color=y)
fig1 = px.scatter_3d(X_top, x=X_top.columns[0], y=X_top.columns[1], z=y, color=y)
# Set the layout and axis labels
fig.update_layout(
    title='3D Scatter plot of the Data respect to top two features',
    scene=dict(
        xaxis_title=top_features_name[0],
        yaxis_title=top_features_name[1],
        zaxis_title='Grounded True value'
    ) 
) 
#fig.show()

# Plot the decision surface 
x_min, x_max = X_top.iloc[:, 0].min() -1, X_top.iloc[:, 0].max() +1 
y_min, y_max = X_top.iloc[:, 1].min() -1, X_top.iloc[:, 1].max() +1  
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))  

# Create Dummy data for reduced feature space 
dummy_X = np.zeros((xx.ravel().shape[0], len(X.columns)))  
dummy_X[:, top_indicies] = np.c_[ xx.ravel(), yy.ravel() ] 

# Predict the output for each point in the mesh grid 
Z = tree_clf.predict( dummy_X )  
Z = Z.reshape(xx.shape) 

# Create the 3D surface plot
fig2 = go.Figure(data=[go.Surface(x=xx, y=yy, z=Z, colorscale='Viridis')])

# Set the layout and axis labels
fig2.update_layout(
    title='Prediction Surface of the Decision Tree',
    scene=dict(
        xaxis_title=top_features_name[0],
        yaxis_title=top_features_name[1],
        zaxis_title='Prediction'
    ) 
)  

# Add the surface plot to the existing scatter plot
fig3 = fig1.add_trace(fig2.data[0])

############################################
# Other models : RF/XGB/SVC
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


