import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def read_data():
    """ Read and split data """

    # Read training data to csv file
    train_data = pd.read_csv('train.csv').drop('Id', axis=1)

    # Get length of data
    data_len = train_data.shape[0]

    x_data = train_data.iloc[:, 1:].to_numpy()
    y_data = train_data['y'].to_numpy()

    return x_data, y_data, data_len


def main():

    fold = 10
    lambdas = np.array([0.01, 0.1, 1, 10, 100])

    x_data, y_data, data_len = read_data()

    R = pd.DataFrame(index=lambdas, columns=range(fold))

    kf = KFold(n_splits=fold)

    for lm in lambdas:

        idx = 0

        for train_idx, test_idx in kf.split(x_data):

            # Split into two groups
            x_test = x_data[test_idx, :]
            y_test = y_data[test_idx]

            x_train = x_data[train_idx, :]
            y_train = y_data[train_idx]

            # Train and compute w
            w = np.linalg.inv(
                x_train.transpose() @ x_train + lm *
                np.eye(x_train.shape[1])) @ x_train.transpose() @ y_train

            # Prediction
            y_pred = w.transpose() @ x_test.transpose()

            # Loss
            R[idx].loc[lm] = np.sqrt(mean_squared_error(y_test, y_pred))
            idx += 1

    # Find mean of each lambda
    R_mean = R.mean(axis=1)

    # Write to file
    R_mean.to_csv('solution.csv', index=False, header=False)


if __name__ == "__main__":
    main()
