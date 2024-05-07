from torch import nn
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn import neural_network
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from os.path import join
import numpy as np

from imbalanced_data.code.neural_network.mlploader import MLPLoader
from imbalanced_data.code.util import utils

import torch
import joblib
import logging
import json

def select_model(config: dict, dataset_name: str, train_method: str, input_size: int, maj_miss_cost: float, min_miss_cost: float, imb_ratio: float):
    model = None
    match train_method:
        case "LogisticRegression":
            model = linear_model.LogisticRegression(random_state=23423) # not used
        case "SVM":
            model = svm.LinearSVC(random_state=42356)
        case "KNearestNeighbours":
            model = neighbors.KNeighborsClassifier()
        case "RandomForest":
            model = ensemble.RandomForestClassifier(random_state=36345)
        case "MLP":
            if config["training"]["MLP"]["pickTorchModel"] == True:
                model = MLPLoader(config, dataset_name, train_method, input_size, maj_miss_cost, min_miss_cost, imb_ratio)
            else:
                model = neural_network.MLPClassifier(random_state=config["training"]["MLP"]["torch_seed"])
        case "MLP-Cost":
            if config["training"]["MLP"]["pickTorchModel"] == True:
                model = MLPLoader(config, dataset_name, train_method, input_size, maj_miss_cost, min_miss_cost, imb_ratio)
            else:
                logging.info("MLP Cost Not available for sklearn mlp")
        case "MLP-Cost-IR":
            if config["training"]["MLP"]["pickTorchModel"] == True:
                model = MLPLoader(config, dataset_name, train_method, input_size, maj_miss_cost, min_miss_cost, imb_ratio)
            else:
                logging.info("MLP Cost Not available for sklearn mlp")
    return model

def training(config: dict, result_dict: dict, start_time: str, current_step: str, experiment_dir: str, dataset_name: str):
    results_dict = {
        "values": {},
        "paths": {}
    }
    results_file_name = f"{current_step}_results_{start_time}"

    train_labels_0 = utils.unpickle(result_dict["preprocessing"]["paths"][f"fold-{0}"]["train_labels"], is_numpy=True)


    for training_method in config["routing"]["training_methods"]:
        # if training_method == "MLP-Cost" and dataset_name != "air_pressure":
        #     logging.warning("MLP-Cost learning method is only available for air_pressure dataset!")
        #     continue

        if training_method == "MLP-Cost-IR" and dataset_name == "air_pressure":
            logging.warning("MLP-Cost-IR learning method is only available for stroke dataset!")
            continue


        results_dict["values"][training_method] = {}
        results_dict["paths"][training_method] = {}

        imb_ratio = None
        maj_miss_cost = None
        min_miss_cost = None
        if dataset_name == "air_pressure" and training_method == "MLP-Cost" :
            pos_cost = 500.0  # minority
            neg_cost = 10.0  # majority
            imb_ratio = pos_cost / neg_cost
            maj_miss_cost = neg_cost / (pos_cost + neg_cost)
            min_miss_cost = pos_cost / (pos_cost + neg_cost)
        if dataset_name == "stroke" and training_method == "MLP-Cost":
            pos_cost = 43100.0 # minority
            neg_cost = 55.0 # majority
            imb_ratio = pos_cost / neg_cost
            maj_miss_cost = neg_cost / (pos_cost + neg_cost)
            min_miss_cost = pos_cost / (pos_cost + neg_cost)
        if dataset_name == "stroke" and training_method == "MLP-Cost-IR":
            pos_count = float(len(train_labels_0[train_labels_0 == 1]))  # minority
            neg_count = float(len(train_labels_0[train_labels_0 == 0]))  # majority
            imb_ratio = neg_count / pos_count
            maj_miss_cost = neg_count / (pos_count + neg_count)
            min_miss_cost = pos_count / (pos_count + neg_count)


        for sampling_method in config["routing"]["sampling_methods"]:
            results_dict["values"][training_method][sampling_method] = {}
            results_dict["paths"][training_method][sampling_method] = {}
            logging.info(f"\n\nStart training with train_method: {training_method} | sampling_method: {sampling_method}")
            for i in range(5):

                # load train features and labels from sampling results
                sampled_features = utils.unpickle(result_dict["sampling"]["paths"][sampling_method][f"fold-{i}"]["train_features"], is_numpy=True)
                sampled_labels = utils.unpickle(result_dict["sampling"]["paths"][sampling_method][f"fold-{i}"]["train_labels"], is_numpy=True)

                model = select_model(config, dataset_name, training_method, len(sampled_features[0]), maj_miss_cost, min_miss_cost, imb_ratio)

                if model is None:
                    break
                logging.info(f"Fold {i}: Train Model")
                # train model
                model.fit(sampled_features, sampled_labels)

                logging.info(f"Fold {i}: Save model")
                model_path = join(experiment_dir, f"{training_method}_{sampling_method}_fold-{i}_model")
                # save model
                if (training_method == "MLP" or training_method == "MLP-Cost" or training_method == "MLP-Cost-IR") and config["training"]["MLP"]["pickTorchModel"] == True:
                    model_path = model_path + ".pth"
                    torch.save(model.get_model().state_dict(), model_path)
                    # model.load_state_dict(torch.load(model_path))
                else:
                    model_path = model_path + ".joblib"
                    joblib.dump(model, model_path)

                # save model paths to result dict
                results_dict["paths"][training_method][sampling_method][f"fold-{i}"] = {}
                results_dict["paths"][training_method][sampling_method][f"fold-{i}"]["model"] = model_path
                utils.save_dict_to_json(results_dict, experiment_dir, results_file_name)

                logging.info(f"Fold {i}: Load Test data and calculate common metrics")
                # load test features and labels from preprocessing
                test_features = utils.unpickle(result_dict["preprocessing"]["paths"][f"fold-{i}"]["test_features"], is_numpy=True)
                y_true = utils.unpickle(result_dict["preprocessing"]["paths"][f"fold-{i}"]["test_labels"], is_numpy=True)

                # predict with test features
                y_pred = model.predict(test_features)

                # calculate common metrics
                sklearn_report = classification_report(y_true, y_pred, output_dict=True)
                imblearn_report = classification_report_imbalanced(y_true, y_pred, output_dict=True)
                imblearn_report["0"]["sup"] = int(imblearn_report["0"]["sup"])
                imblearn_report["1"]["sup"] = int(imblearn_report["1"]["sup"])
                imblearn_report["total_support"] = int(imblearn_report["total_support"])
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                conf_matrix = {"tp": int(tp),
                               "fp": int(fp),
                               "tn": int(tn),
                               "fn": int(fn)}

                # save values to result dict
                results_dict["values"][training_method][sampling_method][f"fold-{i}"] = {}
                results_dict["values"][training_method][sampling_method][f"fold-{i}"]["sklearn_report"] = sklearn_report
                results_dict["values"][training_method][sampling_method][f"fold-{i}"]["imblearn_report"] = imblearn_report
                results_dict["values"][training_method][sampling_method][f"fold-{i}"]["confusion_matrix"] = conf_matrix
                utils.save_dict_to_json(results_dict, experiment_dir, results_file_name)



            if model is None:
                break