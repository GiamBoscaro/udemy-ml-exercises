import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score

# Preprocessing:

dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Preparing the parameter Grid:

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
criterion = ['gini', 'entropy']
max_features = ['auto', 'sqrt', 'log2']
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
min_samples_split = [2, 4, 8, 16]
min_samples_leaf = [2, 4, 8, 16]
bootstrap = [True, False]
 
param_grid = {'n_estimators': n_estimators,
                'criterion': criterion,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

# Finding the best parameters:
 
rf_classifier = RandomForestClassifier()
 
# `n_jobs` means parallel jobs to run -> -1 means using all processors
grid_search = GridSearchCV(rf_classifier, param_grid, cv = 3, verbose = 1, n_jobs = -1)
 
grid_fit = grid_search.fit(X_train, y_train)
grid_fit.best_params_

# Using the best parameters to train the RF classifier:

grid_classifier = RandomForestClassifier(n_estimators=10, 
                                        criterion='gini',
                                        bootstrap=True,
                                        max_depth=10,
                                        max_features='sqrt',
                                        min_samples_leaf=8,
                                        min_samples_split=2,
                                        random_state = 0)
grid_classifier.fit(X_train, y_train)

# Results:

y_pred = rf_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)