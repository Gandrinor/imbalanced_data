import logging
import os
import argparse
import sys
import time

from os.path import split, realpath



available_step_dict = {
    "preprocessing": True,
    "sampling": True,
    "training": True,
    "metric": True
}

previous_step_dict = {
    "sampling": "preprocessing",
    "training": "sampling",
    "metric": "training"
}
def process_step(config: dict, start_time: str, current_step: str, previous_start_time: str, dataset_name: str, verbose: bool):

    if available_step_dict[current_step] != True:
        print("Unrecognized step '%s'", config['step'])
        sys.exit(1)

    experiment_dir = utils.create_experiment_dir(config, start_time, current_step, dataset_name)
    utils.init_logging(config, start_time, current_step, experiment_dir, dataset_name, verbose)
    utils.save_run_and_step_config(config, experiment_dir, current_step, start_time)
    logging.info("Start processing data for dataset {} and : {}\nstart time: {}".format(dataset_name, current_step, start_time))

    if current_step != 'preprocessing':
        previous_experiment_dir = utils.create_experiment_dir(config, previous_start_time, previous_step_dict[current_step], dataset_name)
        utils.copy_config_and_results_from_previous_steps(previous_experiment_dir, experiment_dir)
        result_dict = utils.load_previous_results(experiment_dir)

    match current_step:
        case 'preprocessing':
            if dataset_name == "air_pressure":
                preproair.preprocessing(config, start_time, current_step, experiment_dir, dataset_name)
            elif dataset_name == "stroke":
                preprostroke.preprocessing(config, start_time, current_step, experiment_dir, dataset_name)
            else:
                print(f"preprocessing for dataset {dataset_name} not implemented yet !")
        case 'sampling':
            sample.sampling(config, result_dict, start_time, current_step, experiment_dir, dataset_name)
        case 'training':
            train.training(config, result_dict, start_time, current_step, experiment_dir, dataset_name)
        case 'metric':
            metrics.calcualte_metrics(config, result_dict, start_time, current_step, experiment_dir, dataset_name)

    return start_time


def main():
    root_path = split(realpath(__file__))[0]
    # os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # split(realpath(__file__))[0]
    config = utils.read_json_to_dict(root_path, 'run_config.json')

    arg_parser = argparse.ArgumentParser()
    # Adding arguments
    arg_parser.add_argument('-v', '--verbose', action='store_true', help='Specify if logging outputs should also be printed in stdout')
    # Parsing arguments
    args = arg_parser.parse_args()

    # begin processing steps for each dataset

    datasets: list = config['routing']['data_sets']
    steps: list = config["routing"]["steps"]



    for dataset_name in datasets:
        previous_step_start_time = config["routing"]["previous_step_date"][dataset_name] if config["routing"]["steps"][0] != 'preprocessing' else None
        for step in steps:
            step_start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            previous_step_start_time = process_step(config, step_start_time, step, previous_step_start_time, dataset_name, verbose=args.verbose)



if __name__ == '__main__':
    sys.path.append(os.path.abspath("./../.."))
    print(os.path.abspath("./../.."))
    from imbalanced_data.code.util import utils
    from imbalanced_data.code import preprocessing_air_pressure as preproair
    from imbalanced_data.code import preprocessing_stroke as preprostroke
    from imbalanced_data.code import sample
    from imbalanced_data.code import train
    from imbalanced_data.code import metrics
    main()