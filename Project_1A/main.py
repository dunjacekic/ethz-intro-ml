
#  This task is about using cross-validation for ridge regression. 
#  Remember that ridge regression can be formulated as the following optimization problem ...

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statistics import mean

# Regularization constants specified in task description ---
lambda_regs = [0.01, 0.1, 0.1, 10, 100]

def RMSE(y_pred, y_true):
    return np.sqrt(1/y_pred.shape[0] * np.sum(np.square(y_pred - y_true)))

# Import training data ---
train = np.genfromtxt('./train.csv', delimiter=',', skip_header=True)
train_id = train[:,0]
train_y = train[:,1]
train_x = train[:,2:]

# Standardize the features ---
# Ridge Regression minimizes the hinge loss, while maximizing the margin.
# Margin maximization is achieved through weight norm penalization.
# If features are of different scale, their weights will also be - hence the weight norm penalty will apply differently. 
scaler = StandardScaler()
scaler.fit(train_x)
train_x_std = scaler.transform(train_x)


kf = KFold(n_splits=10)
RMSEs = []
for alpha in lambda_regs:
    losses = []
    
    # For each alpha, train the model, compute RMSE on validation set
    for train_idxes, test_idxes in kf.split(train_x_std):
        X_train, X_test = train_x_std[train_idxes,:], train_x_std[test_idxes,:]
        y_train, y_test = train_y[train_idxes]      , train_y[test_idxes]

        reg = Ridge(alpha)
        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_test)
        losses.append(RMSE(y_pred, y_test))
    
    # Average results over all folds
    RMSEs.append(mean(losses))

print(RMSEs)