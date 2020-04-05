import os

import numpy as np
import pandas as pd
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import SVC


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

    X = pd.DataFrame(columns=new_col)

    # Separate different patients
    for pid in range(int(features['pid'].count() / 12)):

        raw_X = features.iloc[pid * 12: (pid + 1) * 12]

        # Process each column
        new_row = [raw_X['Age'].iloc[0]]
        for col in data_col[3:]:

            # Count and remove NaN
            new_X = raw_X[['Time', col]].dropna()
            real_count = new_X['Time'].count()

            if real_count < 5:

                # If not enough elements
                new_row.extend([np.nan, np.nan, np.mean(
                    new_X[col]), np.std(new_X[col])])

            else:

                # If enough
                Reg = LinearRegression().fit(
                    new_X['Time'].to_numpy().reshape(-1, 1), new_X[col].to_numpy().reshape(-1, 1))

                new_row.extend(
                    [Reg.coef_[0][0], Reg.intercept_[0], np.mean(
                        new_X[col]), np.std(new_X[col])])

        X.loc[pid] = new_row

    return X


def subtask_12(X, test_X, transformer, train_labels, label, N_C, n_splits):
    """ Nested Function for subtask 1 & 2"""

    y = train_labels[label].to_numpy()

    # Initialize variables
    C_list = np.logspace(-2, 2, N_C)
    score_12 = []

    # Cross validation on regularizaiton C
    for C in C_list:

        C_score = []

        # K-Fold split
        kf = KFold(n_splits)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Make pipeline and fit
            clf = make_pipeline(transformer, SVC(
                C=C, kernel='sigmoid', probability=True))
            clf = clf.fit(X_train, y_train)

            # Prediction and scoring
            y_pred = clf.predict_proba(X_test)
            C_score.append(roc_auc_score(y_test, y_pred[:, 1]))

        score_12.append(np.mean(C_score))

    # Find best regularizer
    best_C = C_list[score_12.index(max(score_12))]

    # Fit test data
    clf = make_pipeline(transformer, SVC(
        C=best_C, kernel='sigmoid', probability=True))
    clf = clf.fit(X, y)

    # return clf.predict_proba(test_X)[:, 1]
    return {label: clf.predict_proba(test_X)[:, 1]}


def main():

    train_features, train_labels, test_features = read_data()

    # Columns
    data_col = train_features.columns

    new_col = ['Age']
    for col in data_col[3:]:
        new_col.extend([col + '_w', col + '_b', col + '_mean', col + '_std'])

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

    # Imputations with marking
    transformer = FeatureUnion(transformer_list=[
        ('features', SimpleImputer(strategy='median')),
        ('indicators', MissingIndicator(missing_values=np.nan, features="all"))
    ])

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
            X, test_X, transformer, train_labels, label, N_C, n_splits))

        y_out[label] = y_12

    """ Subtask 3 """
    labels = ['LABEL_RRate', 'LABEL_ABPm',
              'LABEL_SpO2', 'LABEL_Heartrate']
    y = train_labels[labels].to_numpy()

    # Parameters
    n_splits = 5
    N_a = 7

    # Ridge Regression
    reg = make_pipeline(transformer, RidgeCV(
        alphas=np.logspace(-3, 3, N_a), cv=n_splits))
    reg = reg.fit(X, y)
    y_out = pd.concat([y_out, pd.DataFrame(
        data=reg.predict(test_X), columns=labels)], axis=1)

    # Output
    y_out.to_csv('test_labels.zip', index=False,
                 float_format='%.3f', compression='zip')


if __name__ == "__main__":
    main()
