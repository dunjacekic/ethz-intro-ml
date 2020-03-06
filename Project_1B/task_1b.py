import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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

    # read data
    x_data, y_data = read_data()

    # Get all features
    x_data_all = get_features(x_data)

    # Linear regression
    reg = LinearRegression().fit(x_data_all, y_data)

    # Write to file
    np.savetxt('solution.csv', reg.coef_.transpose())
    # reg.coef_.to_csv('solution.csv', index=False, header=False)


if __name__ == "__main__":
    main()
