import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def read_data():
    """ Read and split data """

    # Read training data to csv file
    train_data = pd.read_csv('train.csv').drop('Id', axis=1)

    x_data = train_data.iloc[:, 1:].to_numpy()
    y_data = train_data['y'].to_numpy()

    return x_data, y_data


def main():

    # Constants
    fold = 10
    lambdas = np.array([0.01, 0.1, 1, 10, 100])

    # read data
    x_data, y_data = read_data()

    # Standardize x
    scaler = StandardScaler()
    scaler.fit(x_data)
    x_data = scaler.transform(x_data)

    # Empty loss function
    R = pd.DataFrame(index=lambdas, columns=range(fold))

    # Define folds
    kf = KFold(n_splits=fold)

    # Repeat for each lambda
    for lm in lambdas:

        idx = 0

        # Repeat for different test sets
        for train_idx, test_idx in kf.split(x_data):

            # Split into two groups
            x_test = x_data[test_idx, :]
            y_test = y_data[test_idx]
            x_train = x_data[train_idx, :]
            y_train = y_data[train_idx]

            # Train and compute w
            reg = Ridge(lm)
            reg.fit(x_train, y_train)

            # Prediction
            y_pred = reg.predict(x_test)

            # Loss
            R[idx].loc[lm] = np.sqrt(mean_squared_error(y_test, y_pred))
            idx += 1

    # Find mean of each lambda
    R_mean = R.mean(axis=1)

    # Write to file
    R_mean.to_csv('solution.csv', index=False, header=False)


if __name__ == "__main__":
    main()
