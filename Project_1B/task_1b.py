import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def read_data():
    """ Read and split data """

    # Read training data to csv file
    train_data = pd.read_csv('train.csv').drop('Id', axis=1)

    x_data = train_data.iloc[:, 1:].to_numpy()
    y_data = train_data['y'].to_numpy()

    return x_data, y_data


def get_features(x_data):
    """ Get full feature set """

    x_data_out = np.concatenate(
        [x_data, x_data ** 2, np.exp(x_data), np.cos(x_data),
         np.ones((x_data.shape[0], 1))], axis=1)

    return x_data_out


def main():

    # Constants
    fold = 10
    lambdas = np.logspace(-3, 3)

    # read data
    x_data, y_data = read_data()

    # Get all features
    x_data_all = get_features(x_data)

    # Empty loss function
    R = pd.DataFrame(index=lambdas, columns=range(fold))

    # Set of regression models
    reg_list = []

    # Define folds
    kf = KFold(n_splits=fold)

    # Repeat for each lambda
    for lm in lambdas:

        idx = 0

        # Repeat for different test sets
        for train_idx, test_idx in kf.split(x_data_all):

            # Split into two groups
            x_test = x_data_all[test_idx, :]
            y_test = y_data[test_idx]
            x_train = x_data_all[train_idx, :]
            y_train = y_data[train_idx]

            # Train and compute w
            reg = linear_model.Lasso(lm)
            reg.fit(x_train, y_train)

            # Save coefficients
            reg_list.append(reg.coef_)

            # Prediction
            y_pred = reg.predict(x_test)

            # Loss
            R[idx].loc[lm] = np.sqrt(mean_squared_error(y_test, y_pred))
            idx += 1

    # Find mean of each lambda
    R_mean = R.mean(axis=1)

    # Find minimum of loss
    lm_min = R_mean.idxmin()
    lm_min_idx = np.where(abs(lambdas - lm_min) <= 10 ** -3)[0][0]

    print(lm_min_idx)

    # Write to file
    np.savetxt('solution.csv', reg_list[lm_min_idx].transpose())


if __name__ == "__main__":
    main()
