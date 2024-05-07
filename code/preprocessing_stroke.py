

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from imbalanced_data.code.util import utils

from os.path import join
import json

import logging




def scaling(scaler, train_features, test_features):
    scaler = scaler.fit(train_features)
    return scaler.transform(train_features), scaler.transform(test_features)


def one_hot_encoding(df: pd.DataFrame):
    # create categories list from unique row values

    discrete_columns = df.dtypes[df.dtypes == 'object'].index.to_list()
    print(f"\ndiscrete columns: {discrete_columns}")

    unique_values_count = df[discrete_columns].nunique(axis=0)
    print("\nunique value count for discrete columns:\n{}".format(unique_values_count))
    print("\ntotal rows one hot encoded: {}".format(unique_values_count.sum()))

    categories = []
    print("\nlist unique values of discrete columns:")
    for column in discrete_columns:
        unique_values = pd.unique(df[column]).tolist()
        categories.append(unique_values)
        print(unique_values)

    flat_categories = [unq_val for uniq_vals in categories for unq_val in uniq_vals]
    print("\nflat categories for pd column names: {}".format(flat_categories))

    # encode discrete columns
    enc = OneHotEncoder(sparse_output=False, categories=categories)
    enc.fit(df[discrete_columns])
    result = enc.transform(df[discrete_columns])

    # build dataframe from one hot encoded discrete columns
    one_hot_encoded_discrete_features_df = pd.DataFrame(result, columns=flat_categories)
    # drop old discrete columns
    df = df.drop(labels=discrete_columns, axis=1)
    # add on hot encoded discrete columns to df
    new_df = pd.concat([df, one_hot_encoded_discrete_features_df], axis=1)
    # new_df = df.append(one_hot_encoded_discrete_features_df)

    return new_df


def calculate_train_test_split_statistics(i, labels, train_index, test_index):

    stat_dict = {
        'train': {},
        'test': {}
    }

    selected_labels = labels[train_index]
    pos_count = len(selected_labels[selected_labels == 1])
    neg_count = len(selected_labels[selected_labels == 0])
    stat_dict["train"]["pos_count"] = pos_count
    stat_dict["train"]["neg_count"] = neg_count
    stat_dict["train"]["pos_percentage"] = '%.2f' % (pos_count / (pos_count + neg_count))
    stat_dict["train"]["neg_percentage"] = '%.2f' % (neg_count / (pos_count + neg_count))
    stat_dict["train"]["neg_pos_ratio"] = neg_count / pos_count

    selected_labels = labels[test_index]
    pos_count = len(selected_labels[selected_labels == 1])
    neg_count = len(selected_labels[selected_labels == 0])

    stat_dict["test"]["pos_count"] = pos_count
    stat_dict["test"]["neg_count"] = neg_count
    stat_dict["test"]["pos_percentage"] = '%.2f' % (pos_count / (pos_count + neg_count))
    stat_dict["test"]["neg_percentage"] = '%.2f' % (neg_count / (pos_count + neg_count))
    stat_dict["test"]["neg_pos_ratio"] = neg_count / pos_count

    logging.info(f"\n\nFold {i}:\nTrain / Test data frequencies\n\n{json.dumps(stat_dict, indent=4)}")

    return stat_dict


def preprocessing(config: dict, start_time: str, current_step: str, experiment_dir: str, dataset_name: str):


    results_dict = {
        "values": {},
        "paths": {}
    }
    results_file_name = f"{current_step}_results_{start_time}"
    raw_data_path = join(config["meta"]["data_path"],dataset_name, "00_raw")

    logging.info(f"Loading raw data for dataset: {dataset_name}")
    # load data set
    all_df = pd.read_csv('D:\\Programming\\PycharmProjects\\ba\\imbalanced_data\\data\\stroke\\00_raw\\healthcare-dataset-stroke-data.xls', na_values=["na"])

    logging.info(f"Partially dropping columns")
    # drop rows with NaN bmi values (set index labels to 0, 1, ... n-1 to prevent issues when concatenating one hot encoded discrete features back to array later
    all_df = all_df.dropna(ignore_index=True)
    # drop id row
    all_df = all_df.drop(labels=["id"], axis=1)

    # one hot encode discrete columns
    all_df = one_hot_encoding(all_df)

    # transform data into labels and features in numpy format
    features = all_df.drop(["stroke"], axis=1).values
    labels = all_df["stroke"].to_numpy()

    logging.info("Start splitting data into folds")
    # TODO tqdm -> if using tqdm DONT redirect logging to std output
    # split data into train and test
    sgkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
    split_iterator = sgkf.split(features, labels)
    for i, (train_index, test_index) in enumerate(split_iterator):

        results_dict["paths"][f"fold-{i}"] = {}

        # extract data from indices
        train_features = features[train_index]
        train_labels = labels[train_index]
        test_features = features[test_index]
        test_labels = labels[test_index]

        # calculate train test statistics and save
        stat_dict = calculate_train_test_split_statistics(i, labels, train_index, test_index)
        results_dict["values"][f"fold-{i}"] = stat_dict
        utils.save_dict_to_json(results_dict, experiment_dir, results_file_name)

        logging.info(f"fold {i}: scale data")
        train_features, test_features = scaling(MinMaxScaler(), train_features, test_features)

        # build paths for saving train/ test data
        results_dict["paths"][f"fold-{i}"]["train_features"] = join(experiment_dir, f"fold-{i}_train_features.npy")
        results_dict["paths"][f"fold-{i}"]["train_labels"] = join(experiment_dir, f"fold-{i}_train_labels.npy")
        results_dict["paths"][f"fold-{i}"]["test_features"] = join(experiment_dir, f"fold-{i}_test_features.npy")
        results_dict["paths"][f"fold-{i}"]["test_labels"] = join(experiment_dir, f"fold-{i}_test_labels.npy")

        logging.info(f"fold {i}: save train and test data")
        # save train/ test data
        utils.pickle_save(results_dict["paths"][f"fold-{i}"]["train_features"], train_features, is_numpy=True)
        utils.pickle_save(results_dict["paths"][f"fold-{i}"]["train_labels"], train_labels, is_numpy=True)
        utils.pickle_save(results_dict["paths"][f"fold-{i}"]["test_features"], test_features, is_numpy=True)
        utils.pickle_save(results_dict["paths"][f"fold-{i}"]["test_labels"], test_labels, is_numpy=True)

        # save paths to results json
        utils.save_dict_to_json(results_dict, experiment_dir, results_file_name)
