
#  This task is about using cross-validation for ridge regression. 
#  Remember that ridge regression can be formulated as the following optimization problem ...

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from statistics import mean
import csv

# Regularization constants specified in task description ---
lambda_regs = [1e-4, 1e-3, 1e-2, 1e-1, 0]

def RMSE(y_pred, y_true):
    return np.sqrt(1/y_pred.shape[0] * np.sum(np.square(y_pred - y_true)))

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

# Standardize the features ---
# Ridge Regression minimizes the hinge loss, while maximizing the margin.
# Margin maximization is achieved through weight norm penalization.
# If features are of different scale, their weights will also be - hence the weight norm penalty will apply differently. 
scaler = StandardScaler()

kf = KFold(n_splits=10, shuffle=True)
best_RMSE = 999
best_weights = None
for alpha in lambda_regs:
    losses = []
    
    # For each alpha, train the model, compute RMSE on validation set
    for train_idxes, test_idxes in kf.split(train_x):
        
        X_train, X_test = train_x[train_idxes,:], train_x[test_idxes,:]
        X_train_new, X_test_new = generate_feats(X_train), generate_feats(X_test)
        y_train, y_test = train_y[train_idxes]  , train_y[test_idxes]
        
        scaler.fit(X_train_new)
        train_x_std = scaler.transform(X_train_new)
        test_x_std  = scaler.transform(X_test_new)

        #reg = Ridge(alpha)
        reg = Lasso(alpha)
        reg.fit(train_x_std, y_train)
        #print(reg.coef_)
        
        y_pred = reg.predict(test_x_std)
        losses.append(RMSE(y_pred, y_test))
    
    print(f"lambda={alpha}, RMSE={mean(losses)}")
    # Average results over all folds
    if mean(losses) < best_RMSE:
        best_RMSE = mean(losses)
        best_weights = reg.coef_

print(f"Best RMSE:{best_RMSE}")
print(f"Best weights:{best_weights}")

output_f = open("output_1b.csv",'w')
writer = csv.writer(output_f)
for weight in best_weights:
    writer.writerow([weight])

output_f.close()