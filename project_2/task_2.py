import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import RidgeCV, LogisticRegression


def read_data():
    """ Parse and process data """

    # Train data
    train_features = pd.read_csv('train_features.csv')
    train_labels = pd.read_csv('train_labels.csv')

    # Test data
    test_features = pd.read_csv('test_features.csv')

    return train_features, train_labels, test_features


def process_features(features, data_col, new_col):
    """ Process features, create BoW for each feature """

    features_full = features.copy()
    X = pd.DataFrame(columns=new_col)

    # Get medians of each column for imputation
    median_col = data_col[3:]
    median = pd.DataFrame(columns=median_col)

    new_row = []

    for col in median_col:
        new_row.append(features[col].loc[features[col].notnull()].median())

    median.loc[0] = list(new_row)

    for pid in range(int(features['pid'].count() / 12)):

        row = range(pid * 12, (pid + 1) * 12)

        # Normalize time
        time = features['Time'].iloc[row]
        time = time - time.iloc[0]

        # Impute patient data
        for col in median_col:

            real_count = features[col].iloc[row].count()

            if real_count == 0:

                for i in row:
                    features_full[col].iloc[i] = median[col].to_numpy()

            elif real_count == 1:

                real_idx = features[col].iloc[row].notnull()
                features_full[col].iloc[row] = features[col].iloc[row].to_numpy()[
                    real_idx][0]

            elif real_count < 4:

                real_idx = features[col].iloc[row].notnull()
                x = time.to_numpy()[real_idx]
                y = features[col].iloc[row].to_numpy()[real_idx]
                f = interp1d(x, y, kind='nearest', fill_value='extrapolate')
                features_full[col].iloc[row] = f(range(12))

            else:

                real_idx = features[col].iloc[row].notnull()
                x = time.to_numpy()[real_idx]
                y = features[col].iloc[row].to_numpy()[real_idx]
                f = interp1d(x, y, kind='linear', fill_value='extrapolate')
                features_full[col].iloc[row] = f(range(12))

        new_row = [features['pid'].iloc[row[0]], features['Age'].iloc[row[0]]
                   ] + list(features_full.iloc[row, 3:].to_numpy().flatten())
        X.loc[pid] = new_row

    return X


def subtask_12(X, test_X, train_labels, label, N_C, n_splits):
    """ Nested Function for subtask 1 & 2"""

    y = train_labels[label].to_numpy()

    clf = LogisticRegression(max_iter=50000).fit(X, y)
    print('Score is: %.3f\n' % clf.score(X, y))

    return {label: clf.predict_proba(test_X)[:, 1]}


def main():

    train_features, train_labels, test_features = read_data()

    # Columns
    data_col = train_features.columns

    new_col = ['pid', 'Age']
    for col in data_col[3:]:
        new_col.extend([col + '_' + str(t) for t in range(12)])

    # Process features, cache files for future use
    if 'train_X.csv' in os.listdir() and 'test_X.csv' in os.listdir():

        train_X = pd.read_csv('train_X.csv')
        test_X = pd.read_csv('test_X.csv')

    else:

        train_X = process_features(train_features, data_col, new_col)
        test_X = process_features(test_features, data_col, new_col)
        train_X.to_csv('train_X.csv', index=False)
        test_X.to_csv('test_X.csv', index=False)

    # Remove pid column in training label and testing features
    train_labels = train_labels.iloc[:, 1:]
    test_pid = test_features['pid'].iloc[0::12].to_numpy().astype(int)
    test_features = test_features.iloc[:, 1:]

    # Initialize output data frame
    y_out = pd.DataFrame({'pid': test_pid})

    # Initialize data
    X = train_X.to_numpy()
    test_X = test_X.to_numpy()

    """ Subtask 1 & 2 """
    labels = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
              'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
              'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
              'LABEL_Bilirubin_direct', 'LABEL_EtCO2', 'LABEL_Sepsis']

    # Parameters
    n_splits = 5
    N_C = 5

    for label in labels:
        y_12 = pd.DataFrame(subtask_12(
            X, test_X, train_labels, label, N_C, n_splits))

        y_out[label] = y_12

    """ Subtask 3 """
    labels = ['LABEL_RRate', 'LABEL_ABPm',
              'LABEL_SpO2', 'LABEL_Heartrate']
    y = train_labels[labels].to_numpy()

    # Parameters
    n_splits = 5
    N_a = 7

    # Ridge Regression
    reg = RidgeCV(alphas=np.logspace(-3, 3, N_a), cv=n_splits)
    reg = reg.fit(X, y)
    y_out = pd.concat([y_out, pd.DataFrame(
        data=reg.predict(test_X), columns=labels)], axis=1)

    # Output
    y_out.to_csv('test_labels.zip', index=False,
                 float_format='%.3f', compression='zip')


if __name__ == "__main__":
    main()
