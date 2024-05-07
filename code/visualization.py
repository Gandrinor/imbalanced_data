import pandas as pd
import numpy as np
import sys
import os
import argparse
import logging
import shutil
import plotly.graph_objects as go
from os.path import split, realpath, join, isfile
from os import listdir
from scipy.stats import tukey_hsd
from scipy.stats._result_classes import TukeyHSDResult
import kaleido
from statsmodels.sandbox.stats.multicomp import MultiComparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import torch

import compactletterdisplay
import string

# pip install --upgrade "kaleido==0.1.*"    (downgrade kaleido to work properly)



def tukeyLetters(data, group, sort_asc=True):
    # this if statement can be uncommmented if you don't won't to go furhter with out p<0.05
    # if p_val<0.05:    #If the p value is less than 0.05 it then does the tukey
    mod = MultiComparison(data, group)
    thsd = mod.tukeyhsd()
    # print(mod.tukeyhsd())

    # this is a function to do Piepho method.  AN Alogrithm for a letter based representation of al-pairwise comparisons.
    tot = len(thsd.groupsunique)
    # make an empty dataframe that is a square matrix of size of the groups. #set first column to 1
    df_ltr = pd.DataFrame(np.nan, index=np.arange(tot), columns=np.arange(tot))
    df_ltr.iloc[:, 0] = 1
    count = 0
    df_nms = pd.DataFrame('', index=np.arange(tot), columns=[
        'names'])  # I make a dummy dataframe to put axis labels into.  sd stands for signifcant difference

    for i in np.arange(tot):  # I loop through and make all pairwise comparisons.
        for j in np.arange(i + 1, tot):
            # print('i=',i,'j=',j,thsd.reject[count])
            if thsd.reject[count] == True:
                for cn in np.arange(tot):
                    if df_ltr.iloc[i, cn] == 1 and df_ltr.iloc[
                        j, cn] == 1:  # If the column contains both i and j shift and duplicat
                        df_ltr = pd.concat([df_ltr.iloc[:, :cn + 1], df_ltr.iloc[:, cn + 1:].T.shift().T], axis=1)
                        df_ltr.iloc[:, cn + 1] = df_ltr.iloc[:, cn]
                        df_ltr.iloc[i, cn] = 0
                        df_ltr.iloc[j, cn + 1] = 0
                    # Now we need to check all columns for abosortpion.
                    for cleft in np.arange(len(df_ltr.columns) - 1):
                        for cright in np.arange(cleft + 1, len(df_ltr.columns)):
                            if (df_ltr.iloc[:, cleft].isna()).all() == False and (
                            df_ltr.iloc[:, cright].isna()).all() == False:
                                if (df_ltr.iloc[:, cleft] >= df_ltr.iloc[:, cright]).all() == True:
                                    df_ltr.iloc[:, cright] = 0
                                    df_ltr = pd.concat([df_ltr.iloc[:, :cright], df_ltr.iloc[:, cright:].T.shift(-1).T],
                                                       axis=1)
                                if (df_ltr.iloc[:, cleft] <= df_ltr.iloc[:, cright]).all() == True:
                                    df_ltr.iloc[:, cleft] = 0
                                    df_ltr = pd.concat([df_ltr.iloc[:, :cleft], df_ltr.iloc[:, cleft:].T.shift(-1).T],
                                                       axis=1)

            count += 1

    # I sort so that the first column becomes A
    df_ltr = df_ltr.sort_values(by=list(df_ltr.columns), axis=1, ascending=sort_asc)

    # I assign letters to each column
    for cn in np.arange(len(df_ltr.columns)):
        df_ltr.iloc[:, cn] = df_ltr.iloc[:, cn].replace(1, chr(97 + cn))
        df_ltr.iloc[:, cn] = df_ltr.iloc[:, cn].replace(0, '')
        df_ltr.iloc[:, cn] = df_ltr.iloc[:, cn].replace(np.nan, '')

        # I put all the letters into one string
    df_ltr = df_ltr.astype(str)
    df_ltr.sum(axis=1)
    return df_ltr


# learning_map = {
#     "LogisticRegression": "LR",
#     "SVM": "SVM",
#     "KNearestNeighbours": "KNN",
#     "RandomForest": "RF",
#     "MLP": "MLP",
#     "MLP-Cost": "MLP-Cost"
# }
learning_map = {
    "LogisticRegression": "LR",
    "SVM": "SVM",
    "KNearestNeighbours": "KNN",
    "RandomForest": "RF",
    "MLP": "MLP",
    "MLP-Cost": "MLP-Cost",
    "MLP-Cost-IR" : "MLP-Cost-IR"
}

reverse_learning_map = {
    "LR" : "LogisticRegression",
    "SVM": "SVM",
    "KNN" : "KNearestNeighbours",
    "RF": "RandomForest",
    "MLP": "MLP",
    "MLP-Cost": "MLP-Cost",
    "MLP-Cost-IR" : "MLP-Cost-IR"
}


# metrics_map = {
#     "precision_1": "prec_1",
#     "precision_0": "prec_0",
#     "precision": "prec",
#     "recall_1": "rec_1",
#     "recall_0": "rec_0",
#     "recall": "rec",
#     "accuracy": "acc",
#     "balanced_accuracy": "acc_bal",
#     "geometric-mean": "geo-mean",
#     "cost_function": "cost_fct",
#     "au-roc": "au-roc",
#     "au-prc_1": "au-prc_1",
#     "au-prc_0": "au-prc_1",
#     "au-prc": "au-prc",
#     "average_precision_score_1": "avg_prec_1",
#     "average_precision_score_0": "avg_prec_0",
#     "average_precision_score": "avg_prec",
#     "f1-score_1": "f1_1",
#     "f1-score_0": "f1_0",
#     "f1-score": "f1"
# }
metrics_map = {
    "Cost": "Cost",
    "G-Mean": "G-Mean",
    "F1": "F1",
    "Balanced-Acc":" Bal_Acc",
    "Accuracy": "Acc",
    "Precision": "Pre",
    "Recall" : "Rec",
    "AUROC" : "AUROC",
    "AUPR": "AUPR",
}

abbreviation_map = learning_map

def abbr_or_id(name: str):
    return abbreviation_map.get(name) if abbreviation_map.get(name) is not None else name

def df_to_table(df: pd.DataFrame, arg_max_mask, round_digits: int, fig_width: int, fig_height: int, force_abbreviation: bool):

    fill_colors = "white"
    # fif argmax mask is given, mark best item each column
    if arg_max_mask is not None:
        fill_colors = np.vectorize(lambda x: "lightgrey" if x else "white")(arg_max_mask)

    column_titles = df.columns.values
    if force_abbreviation:
        column_titles = list(map(lambda x: abbr_or_id(x), column_titles))
    column_titles = list(map(lambda x: f"<b>{x}</b>", column_titles))
    column_titles = np.insert(column_titles, 0, "")

    index = df.index.values
    if force_abbreviation:
        index = list(map(lambda x: abbr_or_id(x), index))
    index = list(map(lambda x: f"<b>{x}</b>", index))
    index = np.reshape(index,(len(index), 1))


    cols: list = df.columns.values.tolist()
    if "HSD" in cols:
        cols.remove("HSD")
        hsd_values = [[value] for value in df["HSD"].values.tolist()]
        values = np.round(df[cols].values.astype(np.float64), round_digits)
        extended_values = np.hstack((index, values, hsd_values)).transpose()
    else:
        values = np.round(df[cols].values.astype(np.float64), round_digits)
        extended_values = np.hstack((index, values)).transpose()

    # columnwidth = np.repeat(200, len(column_titles))

    fig = go.Figure(
        data=[go.Table(

            header=dict(
                values=column_titles,
                line_color='black',
                fill_color='white',
                align='center',
                font=dict(
                    color='black',
                    size=10)
            ),
            cells=dict(
                values=extended_values,
                line_color='black',
                fill_color=fill_colors,
                align=['left', 'center'],
                font=dict(
                    color='black',
                    size=9)
            )
        )],
        layout=go.Layout(
            width=fig_width,
            height=fig_height,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    )
    # fig.update_layout(autosize=True)


    # fig.show()
    return fig
    # fig.to_image(format="svg", engine="kaleido")
    # fig.write_image("D:\\Programming\\PycharmProjects\\ba\\imbalanced_data\\code\\table.svg", format="svg", engine="kaleido")
    # pio.write_image(fig, "D:\\Programming\\PycharmProjects\\ba\\imbalanced_data\\code\\table.svg", engine="kaleido")

def calculate_best_arg_map(df: pd.DataFrame, force_smaller_better: bool = False, axis: int = 0):

    if force_smaller_better:
        values = df.values
        index_false_vector = np.full((len(values), 1), False, dtype=bool)
        arg_min_mask = values.min(axis=axis, keepdims=1) == values
        arg_min_mask = np.hstack((index_false_vector, arg_min_mask)).transpose()
        return arg_min_mask

    values = df.values
    index_false_vector = np.full((len(values), 1), False, dtype=bool)
    arg_max_mask = values.max(axis=axis, keepdims=1) == values

    if "Cost" in df.columns:
        cost_index = df.columns.get_loc("Cost")
        cost_values = df.values[:, cost_index]
        cost_arg_min = cost_values.min(axis=0, keepdims=1) == cost_values
        arg_max_mask[:, cost_index] = cost_arg_min

    arg_max_mask = np.hstack((index_false_vector, arg_max_mask)).transpose()
    return arg_max_mask

def metric_mean(metric_list: list, metric_name: str):
    if metric_name != "Cost":
        return np.mean(metric_list)
    else:
        return round(np.mean(metric_list), 2)

# # 11 x vergleiche sampling verfahren (mean über Datensätze und Trainings verfahren) [None Baseline]
# def global_sampling_df(selected: dict, metrics: dict):
#     df = pd.DataFrame(data=None, index=selected["sampling_methods"], columns=selected["metrics"])
#     for sampling_method in selected["sampling_methods"]:
#         for metric in selected["metrics"]:
#             metric_list = []
#             for dataset in selected["data_sets"]:
#                 metric_dict = metrics[dataset]
#                 for training_method in selected["training_methods"]:
#                     metric_list.append(np.mean(metric_dict[training_method][sampling_method][metric]["mean"]))
#             df.loc[sampling_method, metric] = metric_mean(metric_list, metric)
#     return df
#
# # 6x vergleiche trainingsverfahren [cost, decision tree] ( mean über datensatz, sampling verfahren)
# def global_training_df(selected: dict, metrics: dict):
#     df = pd.DataFrame(data=None, index=selected["training_methods"], columns=selected["metrics"])
#     for training_method in selected["training_methods"]:
#         for metric in selected["metrics"]:
#             metric_list = []
#             for dataset in selected["data_sets"]:
#                 metric_dict = metrics[dataset]
#                 for sampling_method in selected["sampling_methods"]:
#                     metric_list.append(np.mean(metric_dict[training_method][sampling_method][metric]["mean"]))
#             df.loc[training_method, metric] = metric_mean(metric_list, metric)
#     return df
#
# # 2x vergleiche datensätze (mean über training, sampling verfahren)
# def global_dataset_df(selected: dict, metrics: dict):
#     df = pd.DataFrame(data=None, index=selected["data_sets"], columns=selected["metrics"])
#     for dataset in selected["data_sets"]:
#         metric_dict = metrics[dataset]
#         for metric in selected["metrics"]:
#             metric_list = []
#             for training_method in selected["training_methods"]:
#                 for sampling_method in selected["sampling_methods"]:
#                     metric_list.append(np.mean(metric_dict[training_method][sampling_method][metric]["mean"]))
#             df.loc[dataset, metric] = metric_mean(metric_list, metric)
#     return df
#
#
# # 66x vergleiche mischung aus trainingsverfahren/ sampling (schneide über datensatz)
# def merged_training_sampling(selected: dict, metrics: dict, selected_metric: str):
#
#     train_sampling_names = []
#     for training_method in selected["training_methods"]:
#         for sampling_method in selected["sampling_methods"]:
#             train_sampling_names.append(f"{abbr_or_id(training_method)} | {abbr_or_id(sampling_method)}")
#
#     df = pd.DataFrame(data=None, index=train_sampling_names, columns=[selected_metric])
#
#     for training_method in selected["training_methods"]:
#         for sampling_method in selected["sampling_methods"]:
#             metric_list = []
#             for dataset in selected["data_sets"]:
#                 metric_dict = metrics[dataset]
#                 metric_list.append(np.mean(metric_dict[training_method][sampling_method][selected_metric]["mean"]))
#             df.loc[f"{abbr_or_id(training_method)} | {abbr_or_id(sampling_method)}", selected_metric] = metric_mean(metric_list, selected_metric)
#     return df
#
# # 12x vergleiche mischung aus trainingsverfahren/ datesatz (schneide über sampling)
# def merged_dataset_training(selected: dict, metrics: dict, selected_metric: str):
#     train_dataset_names = []
#     for training_method in selected["training_methods"]:
#         for dataset in selected["data_sets"]:
#             train_dataset_names.append(f"{abbr_or_id(dataset)} | {abbr_or_id(training_method)}")
#
#     df = pd.DataFrame(data=None, index=train_dataset_names, columns=[selected_metric])
#
#     for training_method in selected["training_methods"]:
#         for dataset in selected["data_sets"]:
#             metric_list = []
#             for sampling_method in selected["sampling_methods"]:
#                 metric_dict = metrics[dataset]
#                 metric_list.append(np.mean(metric_dict[training_method][sampling_method][selected_metric]["mean"]))
#             df.loc[f"{abbr_or_id(dataset)} | {abbr_or_id(training_method)}", selected_metric] = metric_mean(metric_list, selected_metric)
#     return df
#
# # 22x vergleiche mischung aus sampling/ datesatz (schneide über traininsverfahren)
# def merged_dataset_sampling(selected: dict, metrics: dict, selected_metric: str):
#     sampling_dataset_names = []
#     for sampling_method in selected["sampling_methods"]:
#         for dataset in selected["data_sets"]:
#             sampling_dataset_names.append(f"{abbr_or_id(dataset)} | {abbr_or_id(sampling_method)}")
#
#     df = pd.DataFrame(data=None, index=sampling_dataset_names, columns=[selected_metric])
#
#     for sampling_method in selected["sampling_methods"]:
#         for dataset in selected["data_sets"]:
#             metric_list = []
#             for training_method in selected["training_methods"]:
#                 metric_dict = metrics[dataset]
#                 metric_list.append(np.mean(metric_dict[training_method][sampling_method][selected_metric]["mean"]))
#             df.loc[f"{abbr_or_id(dataset)} | {abbr_or_id(sampling_method)}", selected_metric] = metric_mean(metric_list, selected_metric)
#     return df
#
# # (11X6) on matrix per metric comparing all permutations of sampling and training methods
# def single_training_sampling_matrix(selected: dict, metrics: dict, selected_metric: str, selected_dataset: str):
#     metric_dict = metrics[selected_dataset]
#
#     df = pd.DataFrame(data=None, index=selected["training_methods"], columns=selected["sampling_methods"])
#
#     for training_method in selected["training_methods"]:
#         for sampling_method in selected["sampling_methods"]:
#             metric_list = [np.mean(metric_dict[training_method][sampling_method][selected_metric]["mean"])]
#             df.loc[training_method, sampling_method] = metric_mean(metric_list, selected_metric)
#     return df
#
# # (11X6) on matrix per metric comparing all permutations of sampling and training methods
# def single_sampling_training_matrix(selected: dict, metrics: dict, selected_metric: str, selected_dataset: str):
#     metric_dict = metrics[selected_dataset]
#
#     df = pd.DataFrame(data=None, index=selected["sampling_methods"], columns=selected["training_methods"])
#
#     for training_method in selected["training_methods"]:
#         for sampling_method in selected["sampling_methods"]:
#             metric_list = [np.mean(metric_dict[training_method][sampling_method][selected_metric]["mean"])]
#             df.loc[sampling_method, training_method] = metric_mean(metric_list, selected_metric)
#     return df

# # 66x vergleiche mischung aus trainingsverfahren/ sampling
# def single_training_sampling(selected: dict, metrics: dict, selected_metric: str, selected_dataset:str):
#     metric_dict = metrics[selected_dataset]
#
#     train_sampling_names = []
#     for training_method in selected["training_methods"]:
#         for sampling_method in selected["sampling_methods"]:
#             train_sampling_names.append(f"{abbr_or_id(training_method)} | {abbr_or_id(sampling_method)}")
#
#     df = pd.DataFrame(data=None, index=train_sampling_names, columns=[selected_metric])
#
#     for training_method in selected["training_methods"]:
#         if training_method == "MLP-Cost-IR" and selected_dataset == "air_pressure":
#             logging.warning("MLP-Cost-IR learning method is only available for stroke dataset!")
#             continue
#         for sampling_method in selected["sampling_methods"]:
#             metric_list = [np.mean(metric_dict[training_method][sampling_method][selected_metric]["mean"])]
#             df.loc[f"{abbr_or_id(training_method)} | {abbr_or_id(sampling_method)}", selected_metric] = metric_mean(metric_list, selected_metric)
#     return df


# 66x vergleiche mischung aus trainingsverfahren/ sampling
def single_training_sampling(selected: dict, metrics: dict, selected_metric: str, selected_dataset:str, hsd_sort_asc: bool):
    metric_dict = metrics[selected_dataset]

    train_sampling_names = []
    for training_method in selected["training_methods"]:
        if training_method == "MLP-Cost-IR" and selected_dataset == "air_pressure":
            logging.warning("MLP-Cost-IR learning method is only available for stroke dataset!")
            continue
        for sampling_method in selected["sampling_methods"]:
            train_sampling_names.append(f"{abbr_or_id(training_method)} | {abbr_or_id(sampling_method)}")

    # print(train_sampling_names)

    df = pd.DataFrame(data=None, index=train_sampling_names, columns=[selected_metric, "HSD"])

    for training_method in selected["training_methods"]:
        if training_method == "MLP-Cost-IR" and selected_dataset == "air_pressure":
            logging.warning("MLP-Cost-IR learning method is only available for stroke dataset!")
            continue
        for sampling_method in selected["sampling_methods"]:
            metric_list = [np.mean(metric_dict[training_method][sampling_method][selected_metric]["mean"])]
            df.loc[f"{abbr_or_id(training_method)} | {abbr_or_id(sampling_method)}", selected_metric] = metric_mean(metric_list, selected_metric)

    return df


# # 6x vergleiche trainingsverfahren [cost, decision tree] sampling verfahren)
# def single_training(selected: dict, metrics: dict, selected_metric: str, selected_dataset: str):
#     metric_dict = metrics[selected_dataset]
#
#     df = pd.DataFrame(data=None, index=selected["training_methods"], columns=[selected_metric])
#
#     for training_method in selected["training_methods"]:
#         if training_method == "MLP-Cost-IR" and selected_dataset == "air_pressure":
#             logging.warning("MLP-Cost-IR learning method is only available for stroke dataset!")
#             continue
#         metric_list = []
#         for sampling_method in selected["sampling_methods"]:
#             metric_list.append(np.mean(metric_dict[training_method][sampling_method][selected_metric]["mean"]))
#         df.loc[training_method, selected_metric] = metric_mean(metric_list, selected_metric)
#     return df

# 6x vergleiche trainingsverfahren [cost, decision tree] sampling verfahren)


def single_training(selected: dict, metrics: dict, selected_metric: str, selected_dataset: str, hsd_sort_asc: bool):
    metric_dict = metrics[selected_dataset]

    train_methods = selected["training_methods"]
    if selected_dataset == "air_pressure" and "MLP-Cost-IR" in selected["training_methods"]:
        train_methods = [i for i in selected["training_methods"]]
        train_methods.remove("MLP-Cost-IR")

    df = pd.DataFrame(data=None, index=train_methods, columns=[selected_metric, "HSD"])

    for training_method in train_methods:
        if training_method == "MLP-Cost-IR" and selected_dataset == "air_pressure":
            logging.warning("MLP-Cost-IR learning method is only available for stroke dataset!")
            continue
        value = np.mean(metric_dict[training_method]["None"][selected_metric]["mean"])
        df.loc[training_method, selected_metric] = metric_mean(value, selected_metric)



    # HSD calculation preparation
    training_data = []
    training_group = []
    group_nr = 0
    for training_method in train_methods:
        if training_method == "MLP-Cost-IR" and selected_dataset == "air_pressure":
            logging.warning("MLP-Cost-IR learning method is only available for stroke dataset!")
            continue
        for i in range(5):
            value = np.mean(
                metric_dict["single_metrics"][training_method]["None"][f"fold-{i}"][selected_metric])
            training_data.append(value)
            training_group.append(group_nr)
        group_nr += 1


    df_ltr = tukeyLetters(training_data, training_group, hsd_sort_asc)
    letters = df_ltr.sum(axis=1).tolist()
    cap_letters = [letter.upper() for letter in letters]

    counter = 0
    for training_method in train_methods:
        if training_method == "MLP-Cost-IR" and selected_dataset == "air_pressure":
            logging.warning("MLP-Cost-IR learning method is only available for stroke dataset!")
            continue
        df.loc[training_method, "HSD"] = cap_letters[counter]
        counter += 1
    # print(df)

    return df



# 11 x vergleiche sampling verfahren (mean Trainings verfahren)
def single_sampling(selected: dict, metrics: dict, selected_metric: str, selected_dataset: str, hsd_sort_asc: bool):
    metric_dict = metrics[selected_dataset]

    df = pd.DataFrame(data=None, index=selected["sampling_methods"], columns=[selected_metric, "HSD"])

    for sampling_method in selected["sampling_methods"]:
        metric_list = []
        for training_method in selected["training_methods"]:
            if training_method == "MLP-Cost-IR" and selected_dataset == "air_pressure":
                logging.warning("MLP-Cost-IR learning method is only available for stroke dataset!")
                continue
            metric_list.append(np.mean(metric_dict[training_method][sampling_method][selected_metric]["mean"]))
        df.loc[sampling_method, selected_metric] = metric_mean(metric_list, selected_metric)



    # HSD calculation preparation
    sampling_data = []
    sampling_group = []
    group_nr = 0
    for sampling_method in selected["sampling_methods"]:
        metric_list = []
        for training_method in selected["training_methods"]:
            if training_method == "MLP-Cost-IR" and selected_dataset == "air_pressure":
                logging.warning("MLP-Cost-IR learning method is only available for stroke dataset!")
                continue
            for i in range(5):
                value = np.mean(metric_dict["single_metrics"][training_method][sampling_method][f"fold-{i}"][selected_metric])
                metric_list.append(value)

        for element in metric_list:
            sampling_data.append(element)
            sampling_group.append(group_nr)
        group_nr += 1
    df_ltr = tukeyLetters(sampling_data, sampling_group, hsd_sort_asc)
    letters = df_ltr.sum(axis=1).tolist()
    cap_letters = [letter.upper() for letter in letters]

    counter = 0
    for sampling_method in selected["sampling_methods"]:
        df.loc[sampling_method, "HSD"] = cap_letters[counter]
        counter += 1
    # print(df)

    return df

    # print(sampling_group)
    # print(sampling_data)

    # print(cap_letters)

    # print(df_ltr.sum(axis=1))

    # tuk = pairwise_tukeyhsd(sampling_data, sampling_group)
    # print(tuk)
    # tukesLettersAltAlt(tuk)
    # letters = tukeyLetters(tuk.pvalues)

    # hsd = tukey_hsd(*alt_group)
    # print(hsd)

    # df = pd.DataFrame(sample_dict)
    # columns = df.columns.tolist()
    # result_df = compactletterdisplay.anova_cld(data=df, columns=columns, alpha=0.05,method="TukeyHSD")
    # print(result_df)

    # char_list = []
    # for item in result_df['CLD'].tolist():
    #     for c in [char for char in item]:
    #         char_list.append(c)
    # char_list.sort()
    # unique_chars = np.unique(char_list)
    # reverse_map = {}
    # for i, ch in enumerate(unique_chars):
    #     reverse_map[ch] = unique_chars[len(unique_chars)-i-1]
    #
    # new_cld_list= []
    # for item in result_df['CLD'].tolist():
    #     reverse_char_list = []
    #     for c in [char for char in item]:
    #         reverse_char_list.append(reverse_map[c])
    #     reverse_char_list.sort()
    #     new_cld_list.append(''.join(reverse_char_list).upper())
    # result_df['CLD'] = pd.Series(new_cld_list)
    # print(result_df)





# df_creation_map = {
#     "global_dim": {
#       "sampling": global_sampling_df,
#       "training": global_training_df,
#       "dataset": global_dataset_df
#     },
#     "merged_dim": {
#       "dataset_sampling": merged_dataset_sampling,
#       "dataset_training": merged_dataset_training,
#       "training_sampling": merged_training_sampling
#     },
#     "dataset_dim": {
#         "single": {
#             "training_sampling": single_training_sampling,
#             "sampling": single_sampling,
#             "training": single_training
#         },
#         "matrix": {
#             "training_sampling": single_training_sampling_matrix,
#             "sampling_training": single_sampling_training_matrix
#         }
#     }
# }

df_creation_map = {
    "dataset_dim": {
        "single": {
            "training_sampling": single_training_sampling,
            "sampling": single_sampling,
            "training": single_training
        }
    }
}


def get_df_creation_map():
    return df_creation_map

visualize_config_map = {
    "global_dim": {
        "sampling": {
            "round_digits": 3,
            "fig_width": 1920,
            "fig_height": 370,
            "force_abbreviation": True
        },
        "training": {
            "round_digits": 3,
            "fig_width": 1920,
            "fig_height": 240,
            "force_abbreviation": True
        },
        "dataset": {
            "round_digits": 3,
            "fig_width": 1920,
            "fig_height": 130,
            "force_abbreviation": True
        }
    },
    "merged_dim": {
        "dataset_sampling": {
            "round_digits": 3,
            "fig_width": 440,
            "fig_height": 670,
            "force_abbreviation": False
        },
        "dataset_training": {
            "round_digits": 3,
            "fig_width": 360,
            "fig_height": 400,
            "force_abbreviation": False
        },
        "training_sampling": {
            "round_digits": 3,
            "fig_width": 360,
            "fig_height": 1860,
            "force_abbreviation": False
        }
    },
    "dataset_dim": {
        "single": {
            "training_sampling": {
                "round_digits": 3,
                "fig_width": 360,
                "fig_height": 1860,
                "force_abbreviation": True
            },
            "sampling": {
                "round_digits": 3,
                "fig_width": 360,
                "fig_height": 370,
                "force_abbreviation": True
            },
            "training": {
                "round_digits": 3,
                "fig_width": 280,
                "fig_height": 280,
                "force_abbreviation": True
            }
        },
        "matrix": {
            "training_sampling": {
                "round_digits": 3,
                "fig_width": 1300,
                "fig_height": 240,
                "force_abbreviation": True
            },
            "sampling_training": {
                "round_digits": 3,
                "fig_width": 640,
                "fig_height": 370,
                "force_abbreviation": True
            }
        }
    }
}

def get_visualize_config_map():
    return visualize_config_map

def visualization(visualization_config:dict, metrics:dict):

    selected = visualization_config["selection"]
    categories = visualization_config["categories"]
    vis_root = visualization_config["meta"]["visualization_root_dir"]
    vis_file_format = visualization_config["meta"]["file_format"]


    for sub_category in categories["dataset_dim"]["single"]:
        for selected_dataset in selected["data_sets"]:
            for selected_metric in selected["metrics"]:
                print(f"dataset_dim | single | {sub_category} | {selected_dataset} | {selected_metric}")

                df = get_df_creation_map()["dataset_dim"]["single"][sub_category](selected, metrics, selected_metric, selected_dataset, hsd_sort_asc=True) #(True if selected_metric != "Cost" else False)
                df = df.sort_values(selected_metric, axis=0, ascending=(False if selected_metric != "Cost" else True))


                fig = df_to_table(df=(df.drop(columns=['HSD']) if sub_category == "training_sampling" else df), arg_max_mask=None, **get_visualize_config_map()["dataset_dim"]["single"][sub_category])
                fig_save_path = join(vis_root, "dataset_dim", selected_dataset, "single", sub_category, f"all_{selected_metric}.{vis_file_format}")
                if os.path.exists(fig_save_path): os.remove(fig_save_path)
                fig.write_image(fig_save_path, format=vis_file_format, engine="kaleido")





                if sub_category == "training_sampling":

                    # first_six = [*range(12)]
                    # last_six = [*range(df.shape[0] - 12, df.shape[0])]
                    # indices = first_six + last_six
                    # twelve_best_df = df.iloc[indices]

                    data = []
                    group = []
                    group_nr = 0

                    first_twelve = [*range(12)]
                    twelve_best_df = df.iloc[first_twelve]
                    row_names = twelve_best_df.index.values.tolist()
                    metric_dict = metrics[selected_dataset]

                    # HSD calculation preparation
                    for name in row_names:
                        name_arr = name.split("|",1)
                        learner = name_arr[0].strip()
                        sampler = name_arr[1].strip()
                        for i in range(5):
                            value = np.mean(
                                metric_dict["single_metrics"][reverse_learning_map[learner]][sampler][f"fold-{i}"][
                                    selected_metric])
                            data.append(value)
                            group.append(group_nr)
                        group_nr += 1

                    df_ltr = tukeyLetters(data, group, False)
                    letters = df_ltr.sum(axis=1).tolist()
                    cap_letters = [letter.upper() for letter in letters]

                    counter = 0
                    for name in row_names:
                        twelve_best_df.loc[name, "HSD"] = cap_letters[counter]
                        counter += 1
                    # print(twelve_best_df)


                    fig = df_to_table(df=twelve_best_df, arg_max_mask=None, round_digits=3, fig_width=580, fig_height=400, force_abbreviation=True)
                    fig_save_path = join(vis_root, "dataset_dim", selected_dataset, "single", sub_category,f"best_{selected_metric}.{vis_file_format}")
                    if os.path.exists(fig_save_path): os.remove(fig_save_path)
                    fig.write_image(fig_save_path, format=vis_file_format, engine="kaleido")


    # global
    # for sub_category in categories["global_dim"]:
    #     df = get_df_creation_map()["global_dim"][sub_category](selected, metrics)
    #     arg_max_mask = calculate_best_arg_map(df)
    #     fig = df_to_table(df=df, arg_max_mask=arg_max_mask, **get_visualize_config_map()["global_dim"][sub_category])
    #     fig_save_path = join(vis_root, "global_dim", f"{sub_category}.{vis_file_format}")
    #     if os.path.exists(fig_save_path): os.remove(fig_save_path)
    #     fig.write_image(fig_save_path, format=vis_file_format, engine="kaleido")
    #
    # # merged
    # for sub_category in categories["merged_dim"]:
    #     for selected_metric in selected["metrics"]:
    #         print(f"merged_dim | {sub_category} | {selected_metric}")
    #         df = get_df_creation_map()["merged_dim"][sub_category](selected, metrics, selected_metric)
    #         df = df.sort_values(selected_metric, axis=0, ascending=(False if selected_metric != "Cost" else True))
    #         fig = df_to_table(df=df, arg_max_mask=None, **get_visualize_config_map()["merged_dim"][sub_category])
    #         fig_save_path = join(vis_root, "merged_dim", sub_category, f"all_{selected_metric}.{vis_file_format}")
    #         if os.path.exists(fig_save_path): os.remove(fig_save_path)
    #         fig.write_image(fig_save_path, format=vis_file_format, engine="kaleido")
    #
    #         first_six = [*range(6)]
    #         last_six = [*range(df.shape[0] - 6, df.shape[0])]
    #         indices = first_six + last_six
    #         six_best_worst_df = df.iloc[indices]
    #         fig = df_to_table(df=six_best_worst_df, arg_max_mask=None, round_digits=3, fig_width=get_visualize_config_map()["merged_dim"][sub_category]["fig_width"], fig_height=400, force_abbreviation=False)
    #         fig_save_path = join(vis_root, "merged_dim", sub_category, f"best_worst_{selected_metric}.{vis_file_format}")
    #         if os.path.exists(fig_save_path): os.remove(fig_save_path)
    #         fig.write_image(fig_save_path, format=vis_file_format, engine="kaleido")

    # single matrix
    # for sub_category in categories["dataset_dim"]["matrix"]:
    #     for selected_dataset in selected["data_sets"]:
    #         for selected_metric in selected["metrics"]:
    #             print(f"dataset_dim | matrix | {sub_category} | {selected_dataset} | {selected_metric}")
    #
    #             df = get_df_creation_map()["dataset_dim"]["matrix"][sub_category](selected, metrics, selected_metric, selected_dataset)
    #             arg_max_mask = calculate_best_arg_map(df, force_smaller_better=(False if selected_metric != "Cost" else True), axis=0)
    #             fig = df_to_table(df=df, arg_max_mask=arg_max_mask, **get_visualize_config_map()["dataset_dim"]["matrix"][sub_category])
    #             fig_save_path = join(vis_root, "dataset_dim", selected_dataset, "matrix", sub_category, f"{selected_metric}.{vis_file_format}")
    #             if os.path.exists(fig_save_path): os.remove(fig_save_path)
    #             fig.write_image(fig_save_path, format=vis_file_format, engine="kaleido")





    # "dataset_dim": {
    #     "single": [
    #         "training_sampling",
    #         "sampling",
    #         "training"
    #     ],







def main():
    root_path = split(realpath(__file__))[0]
    visualization_config = utils.read_json_to_dict(root_path, 'visualization_config.json')

    arg_parser = argparse.ArgumentParser()
    # Adding arguments
    arg_parser.add_argument('-v', '--verbose', action='store_true',
                            help='Specify if logging outputs should also be printed in stdout')
    # Parsing arguments
    args = arg_parser.parse_args()

    metrics = {}
    for dataset in visualization_config["selection"]["data_sets"]:
        metric_root = join(visualization_config["meta"]["data_path"], dataset, "04_metric", visualization_config["meta"]["previous_metric_step_date"][dataset])
        file_list = [f for f in listdir(metric_root) if isfile(join(metric_root, f))]
        metric_results_list = filter(lambda x: x.find("metric_results") != -1, file_list)
        for metric_result in metric_results_list:
            metrics[dataset] = utils.read_json_to_dict(metric_root, metric_result)

    visualization(visualization_config, metrics)

if __name__ == '__main__':
    sys.path.append(os.path.abspath("./../.."))
    from imbalanced_data.code.util import utils
    main()











# ?10? x 14 = 140 ---> select meaningful ---> special case decision tree
# pro metric !!!
# einen wert pro datensatz, trainingsverfahren, samplingverfahren (132x)





# 1 DIM ranking of methods

# 1x = 6
# 11 x vergleiche sampling verfahren (mean über Datensätze und Trainings verfahren) [None Baseline]
# 6x vergleiche trainingsverfahren [cost, decision tree] ( mean über datensatz, sampling verfahren)
    # 6x eventuell special case, only makes sense in comparison to None case ????

    # 66x vergleiche mischung aus trainingsverfahren/ sampling (schneide über datensatz)
    # 12x vergleiche mischung aus trainingsverfahren/ datesatz (schneide über sampling)
    # 22x vergleiche mischung aus sampling/ datesatz (schneide über traininsverfahren)
    # 132x all combinations

# ---> wenn keine eindeutigen werte -> man kann keine generell aussage ableiten oder mangel in simulation




# 2 DIM methods

# 2 x = 8
# (CHECK)  1x2 2x3 1x3 (the other dim is grouping dimension) in einer reihe der beste/ in einer zeile der beste/ der beste
    # pro Datensatz (11X6)
    # + viele informationen
    # - keine sortierung
# macht 1d ranking sinn ?????
    # 11 x vergleiche sampling verfahren (mean Trainings verfahren)
    # 6x vergleiche trainingsverfahren [cost, decision tree] sampling verfahren)
    # 66x vergleiche mischung aus trainingsverfahren/ sampling


# special cases (cost, decision tree)
# decsion tree vs sampling methods
# cost mlp vs normal mlp (None)

















# TODO for "hard" upsampling/ downsampling methods don't bring classes to euqal level (try 5, 10, 20, 40 %)
#  be aware of influence if sampling methods uses other class examples in its calculation process


# TODO why NOT use best threshold for all nets ??? what is the best treshold ????
#
# big target is simulation study wiht a high degree of comparibalness
# the decision, what is the optimal treshhold, is highly dependent on the current case (for one dataset only)
# it would mean optimizing each case for the highest yield (e.g. lowest cost)
# to optimize this there would be a lot of option (
# hyperparameter tuning,
# one optimal treshhold for 5 and majority vote (ensembles)
# -> calc cost on each net and build mean is not usable for real case, where we have to take a descision
# individual optimal treshhold and majority vote (ensembles)
# if I begin to tweak t for lower cost, it is no longer simulation study and comparable
# TODO maybe use simulation study (general statement / or framework for finding good methods for one use case)
#  for selecting best subset of methods and then try optimization (hyperparameter tuning, ensembles (decision trees), treshhold selection)




