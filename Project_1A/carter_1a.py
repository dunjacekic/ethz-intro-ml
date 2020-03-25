
#  This task is about using cross-validation for ridge regression. 
#  Remember that ridge regression can be formulated as the following optimization problem ...

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statistics import mean
import csv


# Regularization constants specified in task description ---
lambda_regs = [0.01, 0.1, 1, 10, 100]

def RMSE(y_pred, y_true):
    return np.sqrt(1/y_pred.shape[0] * np.sum(np.square(y_pred - y_true)))

# Import training data ---
train = np.genfromtxt('./Project_1A/train.csv', delimiter=',', skip_header=True)
train_id = train[:,0]
train_y = train[:,1]
train_x = train[:,2:]

# Standardize the features ---
# Ridge Regression minimizes the hinge loss, while maximizing the margin.
# Margin maximization is achieved through weight norm penalization.
# If features are of different scale, their weights will also be - hence the weight norm penalty will apply differently. 
scaler = StandardScaler()

kf = KFold(n_splits=10, shuffle=True)
RMSEs = []
csv_f = open('output.csv','w')
writer = csv.writer(csv_f)

for alpha in lambda_regs:
    losses = []
    
    # For each alpha, train the model, compute RMSE on validation set
    for train_idxes, test_idxes in kf.split(train_x):
        
        X_train, X_test = train_x[train_idxes,:], train_x[test_idxes,:]
        y_train, y_test = train_y[train_idxes]  , train_y[test_idxes]
        
        scaler.fit(X_train)
        train_x_std = scaler.transform(X_train)
        test_x_std  = scaler.transform(X_test)

        reg = Ridge(alpha)
        reg.fit(train_x_std, y_train)

        y_pred = reg.predict(test_x_std)
        losses.append(RMSE(y_pred, y_test))
    
    # Average results over all folds
    RMSEs.append(mean(losses))
    writer.writerow([mean(losses)])

print(RMSEs)