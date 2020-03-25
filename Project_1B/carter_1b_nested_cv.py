
#  This task is about using cross-validation for ridge regression. 
#  Remember that ridge regression can be formulated as the following optimization problem ...

import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from statistics import mean
import csv

# Regularization constants specified in task description ---
#param_grid = {
#                'alpha':[1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10, 100]
#             }

param_grid = {
                'alpha':[0.075, 0.05, 0.025, 0.01]
             }

counts = {}
for alpha in param_grid['alpha']:
    counts[alpha] = 0

def RMSE(y_pred, y_true):
    return np.sqrt(1/y_pred.shape[0] * np.sum(np.square(y_pred - y_true)))

def RMSE_score(reg, X_test, y_test):
    return RMSE(reg.predict(X_test), y_test)

def generate_feats(X):
    num_samples = X.shape[0]
    new_feats = np.zeros((num_samples, 21))

    new_feats[:,0:5] = X
    new_feats[:,5:10] = np.square(X)
    new_feats[:,10:15] = np.exp(X)
    new_feats[:,15:20] = np.cos(X)
    new_feats[:,20] = np.ones((num_samples))

    return new_feats
    
# Import training data ---
train = np.genfromtxt('./Project_1B/train.csv', delimiter=',', skip_header=True)
train_id = train[:,0]
train_y = train[:,1]
train_x = train[:,2:]
train_x_new = generate_feats(train_x)

# Standardize the features ---
# Ridge Regression minimizes the hinge loss, while maximizing the margin.
# Margin maximization is achieved through weight norm penalization.
# If features are of different scale, their weights will also be - hence the weight norm penalty will apply differently. 
scaler = StandardScaler()

kf = KFold(n_splits=10, shuffle=True)
best_score = 999
best_reg = None
best_weights = None
regressor = Lasso(max_iter=10000)
trials = 10
for i in range(trials):

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)

    reg = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=inner_cv)
    reg.fit(train_x_new, train_y)

    counts[reg.best_estimator_.alpha] += 1
    nested_score = cross_val_score(reg, X=train_x_new, y=train_y, cv=outer_cv, scoring=RMSE_score)

    print(reg.best_estimator_.coef_)

    if nested_score.mean() < best_score:
        best_score = nested_score.mean()
        best_reg = reg.best_estimator_
        best_weights = reg.best_estimator_.coef_

print(f"Best RMSE:{best_score}")
print(f"Best weights:{best_reg.coef_}")

print(counts)

output_f = open("output_1b.csv",'w')
writer = csv.writer(output_f)
for weight in best_weights:
    writer.writerow([weight])

output_f.close()