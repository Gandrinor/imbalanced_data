from imblearn import over_sampling
from imblearn import under_sampling
from imbalanced_data.code.util import utils

from os.path import join
import json

import logging

def calculate_sampled_data_statistics(sampling_method, i, labels):

    stat_dict = {}

    pos_count = len(labels[labels == 1])
    neg_count = len(labels[labels == 0])
    stat_dict["pos_count"] = pos_count
    stat_dict["neg_count"] = neg_count
    stat_dict["pos_percentage"] = '%.2f' % (pos_count / (pos_count + neg_count))
    stat_dict["neg_percentage"] = '%.2f' % (neg_count / (pos_count + neg_count))
    stat_dict["neg_pos_ratio"] = neg_count / pos_count

    logging.info(f"\n\nSampling Method: {sampling_method} | Fold {i}:\nTrain data frequencies\n\n{json.dumps(stat_dict, indent=4)}")

    return stat_dict

def select_sampler(sampling_method: str, config: dict,):
    sampler = None
    sampler2 = None
    match sampling_method:
        case "None":
            sampler = None
        case "RandomUnder":
            sampler = under_sampling.RandomUnderSampler(random_state=91593)
        case "ENN":
            sampler = under_sampling.EditedNearestNeighbours(n_jobs=-1)
        case "NCR":
            sampler = under_sampling.NeighbourhoodCleaningRule(n_jobs=-1)
        case "TomekLinks":
            sampler = under_sampling.TomekLinks(n_jobs=-1)
        case "RandomOver":
            sampler = over_sampling.RandomOverSampler(random_state=91593,)
        case "SMOTE":
            sampler = over_sampling.SMOTE(random_state=91593, n_jobs=-1)
        case "ADASYS":
            sampler = over_sampling.ADASYN(random_state=91593, n_jobs=-1)
        case "KMeansSMOTE":
            sampler = over_sampling.KMeansSMOTE(random_state=91593, n_jobs=-1, cluster_balance_threshold=config["sampling"]["KMeansSMOTE"]["cluster_balance_threshold"]) # values <= 0.6 are working for stroke
        case "SMOTE-TOMEK":
            sampler = over_sampling.SMOTE(random_state=91593, n_jobs=-1)
            sampler2 = under_sampling.TomekLinks(n_jobs=-1)
        case "SMOTE-ENN":
            sampler = over_sampling.SMOTE(random_state=91593, n_jobs=-1)
            sampler2 = under_sampling.EditedNearestNeighbours(n_jobs=-1)
    return sampler, sampler2
def sampling(config: dict, result_dict: dict, start_time: str, current_step: str, experiment_dir: str, dataset_name: str):

    results_dict = {
        "values": {},
        "paths": {}
    }
    results_file_name = f"{current_step}_results_{start_time}"

    for sampling_method in config["routing"]["sampling_methods"]:
        logging.info(f"Start sampling data with method: {sampling_method}")
        results_dict["values"][sampling_method] = {}
        results_dict["paths"][sampling_method] = {}
        for i in range(5):
            # load data from preprocessing results
            resampled_features = utils.unpickle(result_dict["preprocessing"]["paths"][f"fold-{i}"]["train_features"], is_numpy=True)
            resampled_labels = utils.unpickle(result_dict["preprocessing"]["paths"][f"fold-{i}"]["train_labels"], is_numpy=True)

            # load selected sampling method with fresh random state
            sampler, sampler2 = select_sampler(sampling_method,config)

            # resample data
            logging.info(f"Fold {i}: resample train data")
            if sampler is not None:
                resampled_features, resampled_labels = sampler.fit_resample(resampled_features, resampled_labels)
            if sampler2 is not None:
                resampled_features, resampled_labels = sampler2.fit_resample(resampled_features, resampled_labels)

            # calculate class ratio statistics for sampled data
            stat_dict = calculate_sampled_data_statistics(sampling_method, i, resampled_labels)
            results_dict["values"][sampling_method][f"fold-{i}"] = stat_dict
            utils.save_dict_to_json(results_dict, experiment_dir, results_file_name)

            # build paths for saving resampled train data
            results_dict["paths"][sampling_method][f"fold-{i}"] = {}
            results_dict["paths"][sampling_method][f"fold-{i}"]["train_features"] = join(experiment_dir, f"{sampling_method}_fold-{i}_train_features.npy")
            results_dict["paths"][sampling_method][f"fold-{i}"]["train_labels"] = join(experiment_dir, f"{sampling_method}_fold-{i}_train_labels.npy")

            logging.info(f"Fold {i}: save resampled train data")
            # save train/ test data
            utils.pickle_save(results_dict["paths"][sampling_method][f"fold-{i}"]["train_features"], resampled_features, is_numpy=True)
            utils.pickle_save(results_dict["paths"][sampling_method][f"fold-{i}"]["train_labels"], resampled_labels, is_numpy=True)

            # save paths to results json
            utils.save_dict_to_json(results_dict, experiment_dir, results_file_name)
