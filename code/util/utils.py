import pickle
import logging
import logging.config
import PIL
import numpy as np
from os import path, mkdir, listdir
from os.path import join, exists, split, realpath, isfile, isdir
import os
import json
import shutil
import torch

step_dir_map = { "raw_data": "00_raw",
                 "preprocessing": "01_preprocessed",
                 "sampling": "02_sampled",
                 "training": "03_trained",
                 "metric": "04_metric"}

def pickle_save(file_path, data, is_numpy: bool, force_overwrite=False):
    if path.exists(file_path) and not force_overwrite:
        print('%s already present - Skipping pickling.' % file_path)
    else:
        print('Pickling %s' % file_path)
        try:
            if is_numpy:
                np.save(file_path, data, allow_pickle=True)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to ', file_path, ':', e)


def unpickle(file_path, is_numpy: bool):
    if is_numpy:
        file = np.load(file_path, allow_pickle=True)
    else:
        with open(file_path, 'rb') as f:
            file = pickle.load(f)
            f.close()
    return file
def read_json_to_dict(path, file_name):
    with open(os.path.join(path, file_name), "r") as read_file:
        config = json.load(read_file)
    return config

def save_dict_to_json(dictionary: dict, path: str, file_name: str ):
    path = join(path, f"{file_name}.json")
    with open(path, "w") as f:
        json.dump(dictionary, f, indent=4)

def create_dir(path, name):
    out_dir = join(path, name)
    if not exists(out_dir):
        mkdir(out_dir)
    return out_dir

def create_experiment_dir(config: dict, time: str, step: str, dataset_name ):
    step_dir = join(config["meta"]["data_path"], dataset_name, step_dir_map[step])
    return create_dir(step_dir, time)

def init_logging(config: dict, time: str, current_step: str, experiment_dir:str, dataset_name: str, verbose: bool):
    path = split(realpath(__file__))[0]
    with open(join(path, 'log_config.json')) as json_file:
        log_config_dict = json.load(json_file)

        log_name = '{}_log'.format(current_step)
        log_full_path = join(experiment_dir, '{}.log'.format(log_name))

        log_config_dict['handlers']['file_handler']['filename'] = log_full_path
        log_config_dict['handlers']['screen_handler']['level'] = 'INFO' if verbose else 'WARNING'

        logging.config.dictConfig(log_config_dict)

        # disable external modules' loggers (level warning and below)
        logging.getLogger(PIL.__name__).setLevel(logging.WARNING)

        logging.info("\nStarting Simulation for dataset:{} and steps: {}".format(dataset_name, ", ".join(config["routing"]["steps"])))
        logging.info("\nBegin processing step: {}\nstart time: {}".format(current_step, time))
        logging.info("\nCreated LOG file at: {}".format(log_full_path))

def copy_config_and_results_from_previous_steps(previous_experiment_dir: str, current_experiment_dir: str):
    file_list = [f for f in listdir(previous_experiment_dir) if isfile(join(previous_experiment_dir, f))]
    filtered_file_list = filter(lambda x : x.find("config_") != -1 or x.find("results") != -1, file_list)
    for file in filtered_file_list:
        shutil.copy(join(previous_experiment_dir, file), current_experiment_dir)


def save_run_and_step_config(config: dict, experiment_dir: str, step: str, date: str):

    path = join(experiment_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=4)

    path = join(experiment_dir, 'config_{}_{}.json'.format(step, date))
    with open(path, "w") as f:
        json.dump(config[step], f, indent=4)

def load_previous_results(experiment_dir):
    file_list = [f for f in listdir(experiment_dir) if isfile(join(experiment_dir, f))]
    filtered_file_list = filter(lambda x: x.find("results") != -1, file_list)
    result_dict = {}
    for file in filtered_file_list:
        step_name = file.split("_")[0]
        result_dict[step_name] = {}
        result_dict[step_name]["paths"] = read_json_to_dict(experiment_dir, file)["paths"]
    return result_dict


def init_cuda(device_id, torch_seed):
    torch.random.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if device_id == 'cpu':
            logging.warning('Running on CPU, despite CUDA being available!')
    else:
        device_id = 'cpu'
    device = torch.device(device_id)
    torch.cuda.manual_seed(torch_seed)
    return device


def load_config(experiment_dir: str, step: str):
    file_list = [f for f in listdir(experiment_dir) if isfile(join(experiment_dir, f))]
    filtered_file_list = filter(lambda x: x.find(f"config_{step}") != -1, file_list)
    step_config_dict = read_json_to_dict(experiment_dir, next(filtered_file_list))
    return step_config_dict