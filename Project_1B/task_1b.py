import numpy as np
import pandas as pd
from sklearn import linear_model


def read_data():
    """ Read and split data """

    # Read training data to csv file
    train_data = pd.read_csv('train.csv').drop('Id', axis=1)

    x_data = train_data.iloc[:, 1:].to_numpy()
    y_data = train_data['y'].to_numpy()

    return x_data, y_data


def get_features(x_data):
    """ Get full feature set and standardize """

    x_data_list = [x_data, x_data ** 2, np.exp(x_data), np.cos(x_data),
                   np.ones((x_data.shape[0], 1))]

    x_data_out = np.concatenate(
        x_data_list, axis=1)

    return x_data_out


def main():

    # read data
    x_data, y_data = read_data()

    # Get all features
    x_data_all = get_features(x_data)

    # Ridge regression with built-in cv
    reg = linear_model.RidgeCV(
        alphas=np.logspace(-2, 2, 201),
        cv=5).fit(x_data_all, y_data)

    print(reg.alpha_)

    # Write to file
    np.savetxt('solution.csv', reg.coef_.transpose())


if __name__ == "__main__":
    main()
