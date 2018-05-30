import numpy as np
import pandas as pd
from sklearn import preprocessing

def prep_missing_val(X_train, X_test, y_train, y_test, mode=''):
    print('before prep_missing_val() : ', X_train.shape, X_test.shape)
    if mode == 'mean':
        col_mean = np.nanmean(X_train, axis=0)
        X_train_inds = np.where(np.isnan(X_train))
        X_train[X_train_inds] = np.take(col_mean, X_train_inds[1])
        if X_test != []:
            X_test_inds = np.where(np.isnan(X_test))
            X_test[X_train_inds] = np.take(col_mean, X_test_inds[1])

    elif mode == 'zero':
        X_train = pd.DataFrame(X_train)
        X_train = X_train.fillna(0).values
        if X_test != []:
            X_test = pd.DataFrame(X_test)
            X_test = X_test.fillna(0).values

    elif mode == 'remove':
        X_train_inds = np.where(np.isnan(X_train))
        X_train_inds = list(set(X_train_inds[0]))
        X_train_inds.sort()
        X_train = np.delete(X_train, X_train_inds, axis=0)
        X_train = np.delete(y_train, X_train_inds, axis=0)
        if X_test != []:
            X_test_inds = np.where(np.isnan(X_test))
            X_test_inds = list(set(X_test_inds[0]))
            X_test_inds.sort()
            X_test = np.delete(X_test, X_test_inds, axis=0)
            y_test = np.delete(y_test, X_test_inds, axis=0)
    else:
        print("'mode = %s' is not defined" %(mode))
    print('after prep_missing_val() : ', X_train.shape, y_train.shape)
    return  X_train, X_test, y_train, y_test


def standardization(X_train, X_test=[], mode='zscore', scaler=None):
    new_X_test = []
    if scaler is None:
        if mode == 'zscore':
            scaler = preprocessing.StandardScaler().fit(X_train)
            new_X_train = scaler.transform(X_train)
            if X_test != []:
                new_X_test = scaler.transform(X_test)
        elif mode == 'minmax':
            scaler = preprocessing.MinMaxScaler().fit(X_train)
            new_X_train = scaler.transform(X_train)
            if X_test != []:
                new_X_test = scaler.transform(X_test)
        elif mode == 'quantile':
            scaler = preprocessing.QuantileTransformer(output_distribution='normal').fit(X_train)
            new_X_train = scaler.transform(X_train)
            if X_test != []:
                new_X_test = scaler.transform(X_test)
        elif mode == 'normalize':
            new_X_train = preprocessing.normalize(X_train)
            if X_test != []:
                new_X_test = scaler.transform(X_test)
    else:
        new_X_train = scaler.transform(X_train)
        if X_test != []:
            new_X_test = scaler.transform(X_test)
    new_X_train = new_X_train.astype(np.float32)

    return new_X_train, new_X_test, scaler
