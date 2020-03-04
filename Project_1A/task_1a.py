import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error


def read_data(fold):
    """ Read and split data """

    # Read training data to csv file
    train_data = pd.read_csv('train.csv').drop('Id', axis=1)

    # Get length of data & fold
    data_len = train_data.shape[0]
    fold_len = round(data_len / fold)

    # Split the data into folds
    y_folds = []
    x_folds = []
    for i in range(fold):

        data_range = range(i * fold_len, min((i + 1) * fold_len, data_len))

        y_folds.append(
            train_data['y'].iloc[data_range].to_numpy())
        x_folds.append(train_data.iloc[data_range, 1:].to_numpy())

    return x_folds, y_folds


def main():

    fold = 10
    reps = 10
    lambdas = np.array([0.01, 0.1, 1, 10, 100])

    x_folds, y_folds = read_data(fold)

    R = pd.DataFrame(index=lambdas, columns=range(fold))

    for lm in lambdas:

        for idx in range(reps):

            # Split into two groups
            x_test = x_folds[idx]
            y_test = y_folds[idx]

            x_train = np.concatenate(x_folds[:idx] + x_folds[idx + 1:], axis=0)
            y_train = np.concatenate(y_folds[:idx] + y_folds[idx + 1:], axis=0)

            # Train and compute w
            w = np.linalg.inv(
                x_train.transpose() @ x_train + lm *
                np.eye(x_train.shape[1])) @ x_train.transpose() @ y_train

            # Prediction
            y_pred = w.transpose() @ x_test.transpose()

            # Loss
            R[idx].loc[lm] = np.sqrt(mean_squared_error(y_test, y_pred))

    # Find mean of each lambda
    R_mean = R.mean(axis=1)

    # Write to file
    R_mean.to_csv('solution.csv', index=False, header=False)


if __name__ == "__main__":
    main()
