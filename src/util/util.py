import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

def time_measure(orginal_function):
    def wrapper_function(*args, **kwargs):
        start_time = time.time()
        result = orginal_function(*args, **kwargs)
        return result
    return wrapper_function


def _get_label_encoder(encoder_path):
    if os.path.exists(encoder_path):
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
            return encoder
    else:
        return None


def _make_label_encoder(encoder_path, values):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(values)
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    return encoder


def _get_label_arr(data_df, label_col):
    label_df = data_df[label_col]
    encoder_path = './col_%s.pkl'
    encoder = _get_label_encoder(encoder_path)
    if encoder is None:
        encoder = _make_label_encoder(encoder_path, label_df.values)
    label_arr = encoder.transform(label_df.values)
    print(encoder.classes_)
    return label_arr


def load_data(file_path, num_cols, cate_cols, label_col='', csv_sep=',', **kwargs):
    # 파일 읽기
    data_df = pd.read_csv(file_path, csv_sep)

    num_df = None
    if len(num_cols) is not 0:
        num_df = data_df[num_cols]

    cate_df = None
    if len(cate_cols) is not 0:
        cate_df = data_df[cate_cols]
        cate_df = pd.get_dummies(cate_df, columns=cate_cols)

    if num_df is None and cate_df is None:
        print("비어있습니다")
        exit()
    elif num_df is None and cate_df is not None:
        parsed_data_df = cate_df
    elif num_df is not None and cate_df is None:
        parsed_data_df = num_df
    elif num_df is not None and cate_df is not None:
        parsed_data_df = pd.concat([num_df, cate_df], axis=1)

    data = parsed_data_df.values
    data = data.astype(np.float32)

    if label_col == '':
        return data
    else:
        labels = _get_label_arr(data_df, label_col)
        labels = labels.astype(np.int32)
        return data, labels

def load_data_with_name(file_path, num_cols, cate_cols, label_col='', csv_sep=',', **kwargs):
    # 파일 읽기
    data_df = pd.read_csv(file_path, csv_sep)

    num_df = None
    if len(num_cols) is not 0:
        num_df = data_df[num_cols]

    cate_df = None
    if len(cate_cols) is not 0:
        cate_df = data_df[cate_cols]
        cate_df = pd.get_dummies(cate_df, columns=cate_cols)

    if num_df is None and cate_df is None:
        print("비어있습니다")
        exit()
    elif num_df is None and cate_df is not None:
        parsed_data_df = cate_df
    elif num_df is not None and cate_df is None:
        parsed_data_df = num_df
    elif num_df is not None and cate_df is not None:
        parsed_data_df = pd.concat([num_df, cate_df], axis=1)

    data = parsed_data_df.values
    data = data.astype(np.float32)

    if label_col == '':
        return data, parsed_data_df.columns
    else:
        labels = _get_label_arr(data_df, label_col)
        labels = labels.astype(np.int32)
        label_names = [str(label) for label in list(set(labels))]
        data, labels = shuffle(data, labels, random_state=1)
        return data, labels, parsed_data_df.columns.tolist(), label_names


def load_data_balance(file_path, num_cols, cate_cols, label_col='', csv_sep=',', **kwargs):
    data_arr, labels_arr, feature_names, label_names = load_data_with_name(file_path, num_cols, cate_cols, label_col='', csv_sep=csv_sep)
    N_data_arr = []
    Y_data_arr = []

    for idx, data in enumerate(data_arr):
        if labels_arr[idx] == 0:
            N_data_arr.append(data)
        elif labels_arr[idx] == 1:
            Y_data_arr.append(data)
    N_data_arr = np.array(N_data_arr)
    Y_data_arr = np.array(Y_data_arr)
    n_fold = 0
    data_list = []
    labels_list = []

    if N_data_arr.shape[0] > Y_data_arr.shape[0]:
        n_split = int(N_data_arr.shape[0] / Y_data_arr[0])
        print('n_split', n_split)
        if n_split == 1:
            data_len = Y_data_arr.shape[0]
            N_labels = np.zeros((data_len), dtype=np.int32)
            Y_labels = np.ones((data_len), dtype=np.int32)
            data = np.append(N_data_arr[:data_len, :], Y_data_arr, axis=0)
            labels  = np.append(N_labels, Y_labels)
            data, labels = shuffle(data, labels, random_state=1)
            print(data.shape, labels.shape)
            data_list.append(data)
            labels_list.append(labels)
        else:
            kf = KFold(n_splits=n_split, random_state=1, shuffle=True)
            for train_index, test_index in kf.split(N_data_arr):
                n_fold += 1
                N_labels = np.zeros((N_data_arr[test_index].shape[0]), dtype=np.int32)
                Y_labels = np.ones((Y_data_arr.shape[0]), dtype=np.int32)
                data = np.append(N_data_arr[test_index], Y_data_arr, axis=0)
                labels = np.append(N_labels, Y_labels)
                data, labels = shuffle(data, labels, random_state=1)
                print(data.shape, labels.shape)
                data_list.append(data)
                labels_list.append(labels)
    else:
        n_split = int(N_data_arr.shape[0] / Y_data_arr[0])
        print('n_split', n_split)
        kf = KFold(n_splits=n_split, random_state=1, shuffle=True)
        for train_index, test_index in kf.split(Y_data_arr):
            n_fold += 1
            N_labels = np.zeros((N_data_arr.shape[0]), dtype=np.int32)
            Y_labels = np.ones((Y_data_arr[test_index].shape[0]), dtype=np.int32)
            data = np.append(N_data_arr, Y_data_arr[test_index], axis=0)
            labels = np.append(N_labels, Y_labels)
            data, labels = shuffle(data, labels, random_state=1)
            print(data.shape, labels.shape)
            data_list.append(data)
            labels_list.append(labels)

    return data_list, labels_list, feature_names, label_names