from imbalanced_data.code.util import utils
from imbalanced_data.code.neural_network.mlploader import MLPLoader
from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             confusion_matrix,
                             roc_auc_score, auc,
                             precision_recall_curve,
                             average_precision_score,
                             f1_score,
                             precision_score,
                             recall_score)
from imblearn.metrics import geometric_mean_score

import numpy as np
import logging
import torch
import joblib
import json


def calcualte_single_metric(metric: str, y_test_true, y_test_pred, y_test_pred_prob, dataset_name: str):
    
    match metric:
        case "Accuracy":
            return accuracy_score(y_test_true, y_test_pred)
        case "Balanced-Acc":
            return balanced_accuracy_score(y_test_true, y_test_pred)
        case "G-Mean":
            return geometric_mean_score(y_test_true, y_test_pred)
        case "Cost":
            if dataset_name == "air_pressure":
                # false positive -> truck being checked, but does not have failure in APS -> cost = 10
                # false negative -> truck with failure in APS was not checked -> cost = 500
                tn, fp, fn, tp = confusion_matrix(y_test_true, y_test_pred).ravel()
                return fp * 10 + fn * 500
            if dataset_name == "stroke":
                # false positive -> predicted stroke, but nothing was wrong 10
                # false negative -> stroke was not predicted, but occurred neg_pos_ration * 10 ~ 210
                tn, fp, fn, tp = confusion_matrix(y_test_true, y_test_pred).ravel()
                return fp * 55 + fn * 43100
            logging.warning("No Cost function implemented for dataset {} !".format(dataset_name))
            return None
        case "AUROC":
            return roc_auc_score(y_test_true, y_test_pred_prob)
        case "AUPR":
            precision, recall, thresholds = precision_recall_curve(y_test_true, y_test_pred_prob, pos_label=1)
            return auc(recall, precision)
        case "F1":
            return f1_score(y_test_true, y_test_pred, pos_label=1)
        case "Precision":
            return precision_score(y_test_true, y_test_pred, pos_label=1)
        case "Recall":
            return recall_score(y_test_true, y_test_pred, pos_label=1)
        # case "au-prc_1":
        #     precision, recall, thresholds = precision_recall_curve(y_test_true, y_test_pred_prob, pos_label=1)
        #     return auc(recall, precision)
        # case "au-prc_0":
        #     precision, recall, thresholds = precision_recall_curve(y_test_true, y_test_pred_prob, pos_label=0)
        #     return auc(recall, precision)
        # case "average_precision_score_1":
        #     return average_precision_score(y_test_true, y_test_pred_prob, pos_label=1)
        # case "average_precision_score_0":
        #     return average_precision_score(y_test_true, y_test_pred_prob, pos_label=0)
        # case "f1-score_1":
        #     return f1_score(y_test_true, y_test_pred, pos_label=1)
        # case "f1-score_0":
        #     return f1_score(y_test_true, y_test_pred, pos_label=0)
        # case "precision_1":
        #     return precision_score(y_test_true, y_test_pred, pos_label=1)
        # case "precision_0":
        #     return precision_score(y_test_true, y_test_pred, pos_label=0)
        # case "recall_1":
        #       return recall_score(y_test_true, y_test_pred, pos_label=1)
        # case "recall_0":
        #     return recall_score(y_test_true, y_test_pred, pos_label=0)
        case _:
            logging.warning(f"Selected metric: \"{metric}\" is not supported! Also check for typo!")




def calcualte_metrics(config: dict, result_dict: dict, start_time: str, current_step: str, experiment_dir: str, dataset_name: str):

    single_metric_dict = {}
    metric_dict = {}
    metric_dict["single_metrics"] = {}
    results_file_name = f"{current_step}_results_{start_time}"

    # modify config with actual training config to initialize model (MLP, MLP-Cost) with correct values
    config["training"] = utils.load_config(experiment_dir, "training")

    # TODO save inputs_size, labels ... together
    train_labels_0 = utils.unpickle(result_dict["preprocessing"]["paths"][f"fold-{0}"]["train_labels"], is_numpy=True)
    pos_count = float(len(train_labels_0[train_labels_0 == 1]))  # minority
    neg_count = float(len(train_labels_0[train_labels_0 == 0]))  # majority
    neg_pos_ratio = neg_count / pos_count


    # air pressure mlp cost
    pos_cost_air = 500.0  # minority
    neg_cost_air = 10.0  # majority
    imb_ratio_air = pos_cost_air / neg_cost_air
    maj_miss_cost_air = neg_cost_air / (pos_cost_air + neg_cost_air)
    min_miss_cost_air = pos_cost_air / (pos_cost_air + neg_cost_air)

    # stroke mlp cost
    pos_cost_stroke = 43100.0  # minority
    neg_cost_stroke = 55.0  # majority
    imb_ratio_stroke = pos_cost_stroke / neg_cost_stroke
    maj_miss_cost_stroke = neg_cost_stroke / (pos_cost_stroke + neg_cost_stroke)
    min_miss_cost_stroke = pos_cost_stroke / (pos_cost_stroke + neg_cost_stroke)

    # stroke mlp cost ir
    pos_count = float(len(train_labels_0[train_labels_0 == 1]))  # minority
    neg_count = float(len(train_labels_0[train_labels_0 == 0]))  # majority
    imb_ratio = neg_count / pos_count
    maj_miss_cost = neg_count / (pos_count + neg_count)
    min_miss_cost = pos_count / (pos_count + neg_count)


    mlp_model_map = {
        "stroke":{
            "MLP": MLPLoader(config, dataset_name, "MLP", input_size=21, maj_miss_cost=None, min_miss_cost=None, imb_ratio=None),
            "MLP-Cost": MLPLoader(config, dataset_name, "MLP-Cost", input_size=21, maj_miss_cost=maj_miss_cost_stroke, min_miss_cost=min_miss_cost_stroke , imb_ratio=imb_ratio_stroke),
            "MLP-Cost-IR": MLPLoader(config, dataset_name, "MLP-Cost", input_size=21, maj_miss_cost=maj_miss_cost, min_miss_cost=min_miss_cost , imb_ratio=imb_ratio)

        },
        "air_pressure": {
            "MLP": MLPLoader(config, dataset_name, "MLP", input_size=161, maj_miss_cost=None, min_miss_cost=None, imb_ratio=None),
            "MLP-Cost": MLPLoader(config, dataset_name, "MLP-Cost", input_size=161, maj_miss_cost=maj_miss_cost_air, min_miss_cost=min_miss_cost_air , imb_ratio=imb_ratio_air)
        }
    }


    # traverse selected dataset, training methods, sampling methods, 5 folds
    for training_method in config["routing"]["training_methods"]:

        if training_method == "MLP-Cost-IR" and dataset_name == "air_pressure":
            logging.warning("MLP-Cost-IR learning method is only available for stroke dataset!")
            continue

        single_metric_dict[training_method] = {}
        metric_dict[training_method] = {}
        for sampling_method in config["routing"]["sampling_methods"]:
            single_metric_dict[training_method][sampling_method] = {}
            metric_dict[training_method][sampling_method] = {}
            logging.info(f"\n\nStart calculating metrics for train_method: {training_method} | sampling_method: {sampling_method}")
            for i in range(5):
                single_metric_dict[training_method][sampling_method][f"fold-{i}"] = {}
                model = None
                # load test features and labels from preprocessing
                x_test = utils.unpickle(result_dict["preprocessing"]["paths"][f"fold-{i}"]["test_features"], is_numpy=True)
                y_test_true = utils.unpickle(result_dict["preprocessing"]["paths"][f"fold-{i}"]["test_labels"],is_numpy=True)

                # load model
                model_path = result_dict["training"]["paths"][training_method][sampling_method][f"fold-{i}"]["model"]
                if training_method == "MLP" or training_method == "MLP-Cost" or training_method == "MLP-Cost-IR":
                    model = mlp_model_map[dataset_name][training_method]
                    # TODO no, this negates bug?, where mlp cost state was saved as .joblib
                    state_dict = torch.load(model_path)
                    model.load_state_dict(state_dict)
                else:
                    model = joblib.load(model_path)

                # calc pred treshhold 0.5
                y_test_pred = model.predict(x_test)

                # calculate positive class prediction (sums up to one with negative probability)
                if training_method != "SVM":
                    y_test_pred_prob = model.predict_proba(x_test)[:, 1]
                else:
                    y_test_pred_prob = y_test_pred

                for metric in config["routing"]["metrics"]["single"]:

                    # calculate metric
                    value = calcualte_single_metric(metric, y_test_true, y_test_pred, y_test_pred_prob, dataset_name, neg_pos_ratio)
                    # if metric is not available, skip
                    if value == None:
                        continue
                    single_metric_dict[training_method][sampling_method][f"fold-{i}"][metric] = float(value)

                # print(json.dumps(single_metric_dict, indent=4))

                # for metric in config["routing"]["metrics"]["macro_avg"]:
                #     value_1 = single_metric_dict[training_method][sampling_method][f"fold-{i}"][metric + "_1"]
                #     value_0 = single_metric_dict[training_method][sampling_method][f"fold-{i}"][metric + "_0"]
                #     value = (value_0 + value_1) / 2.0
                #     single_metric_dict[training_method][sampling_method][f"fold-{i}"][metric] = float(value)

            for metric in config["routing"]["metrics"]["single"] + config["routing"]["metrics"]["macro_avg"]:
                if metric in single_metric_dict[training_method][sampling_method][f"fold-{0}"]:
                    # append metrics from single metric dict to list
                    value_list = [single_metric_dict[training_method][sampling_method][f"fold-{j}"][metric] for j in range(5)]

                    # calculate mean and std over folds and save to dict
                    metric_dict[training_method][sampling_method][metric] = {}
                    metric_dict[training_method][sampling_method][metric]["mean"] = np.mean(value_list)
                    metric_dict[training_method][sampling_method][metric]["std"] = np.std(value_list)

            # TODO save single metric dict somewhere inside metric results -> t test oder HSD
            # single_metric_dict[training_method][sampling_method][f"fold-{i}"][metric]
            metric_dict["single_metrics"] = single_metric_dict

            # save metric dict
            utils.save_dict_to_json(metric_dict, experiment_dir, results_file_name)








# on 0,5 threshold prediciton -> 0,1

    # (6) for 0 and 1 respective
        #  precision
        #  recall (positive class case: sensitivity | negative class case: specificity !)
        #  f1-score

    # (3) macro avg (equally weighted metric from 0 and 1 case (0.5 each)) --> does not favour any class
    # avg ?????? -> how to choose weight ????

    # accuracy
    # g mean

# on prob prediction -> 0,1
    # au roc ...
    # au pr 9711 Learning from Imbalanced Data.pdf  -> average_precision_score NO is different from au pr !!! better?



# weighted avg (support weighted metric from 0 and 1 case ) ----> highly favours majority class
# IS DEEMED NOT UNSUITABLE !!!! 1241 (overview, METRICS, everything else DETAILED )overview A Survey of Predictive Modeling on Imbalanced Domains

## inverse class imbalance -> highly favours minority class





