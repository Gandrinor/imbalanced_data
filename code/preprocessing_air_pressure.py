

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from imbalanced_data.code.util import utils

from os.path import join
import json

import logging




def scaling(scaler, train_features, test_features):
    scaler = scaler.fit(train_features)
    return scaler.transform(train_features), scaler.transform(test_features)

def imputing(imputer, train_features, test_features):
    logging.info("fit imputer with train data")
    imputer = imputer.fit(train_features)
    # train ~ 11 min (4/5)n ^2 = 16/25 n2
    logging.info("impute train data")
    train_transform = imputer.transform(train_features)
    # test ~ 2.4 (4/5)n * 1/5n) = 4/25 n2
    logging.info("impute test data")
    test_transform = imputer.transform(test_features)
    return train_transform, test_transform

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
        "values" : {},
        "paths" : {}
    }
    results_file_name = f"{current_step}_results_{start_time}"
    raw_data_path = join(config["meta"]["data_path"],dataset_name, "00_raw")

    logging.info(f"Loading raw data for dataset: {dataset_name}")
    # load data set
    train_df = pd.read_csv(join(raw_data_path, 'aps_failure_training_set.csv'), na_values=["na"])
    test_df = pd.read_csv(join(raw_data_path, 'aps_failure_test_set.csv'), na_values=["na"])

    # merge initial train and test data set
    all_df = pd.concat([train_df, test_df], axis=0)

    logging.info(f"Partially dropping columns")
    # drop columns with too many missing values > 50%
    nan_values = all_df.isna().sum(axis=0).div(all_df.shape[0], axis=0).mul(100).sort_values(ascending=False)
    features_to_be_dropped = nan_values[nan_values > 50].index
    all_df = all_df.drop(labels=features_to_be_dropped, axis=1)

    # drop feature with constant  value
    plain_features = all_df.drop(["class"], axis=1)
    variance = plain_features.var(axis=0)
    all_df = all_df.drop(variance[variance == 0].index, axis=1)

    # transform data into labels and features in numpy format
    features = all_df.drop(["class"], axis=1).values
    labels = all_df["class"].transform(lambda x: 1 if x=="pos" else 0).to_numpy()

    # TODO log split into folds
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

        logging.info(f"fold {i}: impute data")
        train_features, test_features = imputing(KNNImputer(n_neighbors=5, weights="distance", metric="nan_euclidean"), train_features, test_features)

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
