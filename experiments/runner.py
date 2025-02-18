import tensorflow as tf
import os

# if "--gpu" not in sys.argv:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#     tf.config.set_soft_device_placement(True)


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_soft_device_placement(True)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)
# print("GPUs used: ", gpus)
tf.config.run_functions_eagerly(False)

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', 'ns_lib'))
import copy
import argparse
from itertools import product
import wandb
from train import main as main_train
import ns_lib as ns
import ast
import numpy as np
import yaml


class ExperimentConfig:
    """Central configuration class for experiment parameters"""
    def __init__(self):
        
        self.default = self.load_config_from_file(os.path.join(current_dir,'config.yaml')) # load config from json file
 
        self.hparams = {
                'dataset_name': ['countries_s1'],
                'grounder': ['backward_0_1'],
                'model_name': ['dcr'],
                'kge': ['complex'],
                'seed': [[0,1,2,3,4]],
                'epochs': [100], # [100]
                'batch_size': [256],
                'val_batch_size': [256],
                'test_batch_size': [256],
                'resnet': [True], #[True]
                'store_ranks': [False], #[False]
                'stop_kge_gradients': [False],

                'use_logger': [True],
                'use_WB': [False],
                'load_model_ckpt': [False], # [False],
                'load_kge_ckpt': [False],
                'save_model_ckpt': [True],
                'save_kge_ckpt': [True],
                'log_folder': ["./experiments/runs/"], #["./experiments/runs/"],
                'ckpt_folder': ["./../checkpoints/"], #["./../checkpoints/"],
                'data_path': ["experiments/data"],
                'rules_file' : ['rules.txt'], 
        }

        # update default with hparams
        self.config_data = {**self.default, **self.hparams}

        # Initialize from config_data
        for key, val in self.config_data.items():
            setattr(self, key, val)
        
        # delete default values
        del self.hparams
        del self.default
        del self.config_data
        
        # Parse command line arguments
        self.parse_args()

    def load_config_from_file(self, config_file):
        """Load configuration from a YAML file if it exists."""
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) # Use yaml.safe_load for security
                if config_data is None: # Handle empty YAML file
                    config_data = {}
                for key, val in config_data.items():
                        config_data[key] = [val]
        except FileNotFoundError:
            print(f"Config file {config_file} not found. Using default settings from code.")
        except yaml.YAMLError as e: # Catch YAML parsing errors
            print(f"Error parsing YAML from {config_file}: {e}")
        return config_data

    def parse_args(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description='Experiment Runner')
        parser.add_argument("--d", nargs='+', help="Datasets")
        parser.add_argument("--m", nargs='+', help="Models")
        parser.add_argument("--g", nargs='+', help="Grounders")
        parser.add_argument("--s", help="Seeds")
        parser.add_argument("--load_model_ckpt", default = None, help="load model ckpt")
        parser.add_argument("--load_kge_ckpt", default = None, help="load kge ckpt")
        parser.add_argument("--save_model_ckpt", default = None, help="save_model_ckpt")
        parser.add_argument("--log_folder", default = None, help="log folder")
        parser.add_argument("--ckpt_folder", default = None, help="ckpt folder")
        parser.add_argument("--resnet", default = None, help="reset")
        parser.add_argument("--store_ranks", default = None, help="reset")
        parser.add_argument("--epochs", default = None, help="epochs")
        parser.add_argument("--stop_kge_gradients", default = None, help="stop_kge_gradients")
        parser.add_argument("--rules_file", default = None, help="stop_kge_gradients")
        parser.add_argument("--xkge", default = None, help="xkge")

        
        args = parser.parse_args()

        # Update configuration with command line arguments
        if args.d: self.dataset_name = args.d
        if args.m: self.model_name = args.m
        if args.g: self.grounder = args.g
        if args.s: self.seed = [ast.literal_eval(args.s)]
        if args.log_folder: self.log_folder = [args.log_folder]
        if args.ckpt_folder: self.ckpt_folder = [args.ckpt_folder]
        if args.epochs: self.epochs = [int(args.epochs)]
        if args.load_model_ckpt: self.load_model_ckpt = [ast.literal_eval(args.load_model_ckpt)]
        if args.load_kge_ckpt: self.load_kge_ckpt = [ast.literal_eval(args.load_kge_ckpt)]
        if args.save_model_ckpt: self.save_model_ckpt = [ast.literal_eval(args.save_model_ckpt)]
        if args.stop_kge_gradients: self.stop_kge_gradients = [ast.literal_eval(args.stop_kge_gradients)]
        if args.resnet: self.resnet = [ast.literal_eval(args.resnet)]
        if args.store_ranks: self.store_ranks = [ast.literal_eval(args.store_ranks)]
        if args.rules_file: self.rules_file = [args.rules_file]
        if args.xkge is not None and ast.literal_eval(args.xkge): 
            self.resnet = [False]
            self.store_ranks = [True]
            self.log_folder = ["./experiments/runs_xkge/"]
            self.ckpt_folder = ["./../checkpoints_xkge/"]


def setup_tf():
    """Configure TensorFlow settings"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    print("GPUs used: ", gpus)
    tf.config.run_functions_eagerly(False)


def generate_experiments(config):
    """Generate experiment configurations."""
    from experiments.update_config import update_config
    param_names = [attr for attr in dir(config) if not attr.startswith('_') and not callable(getattr(config, attr))]
    params = [getattr(config, name) for name in param_names]
    experiments = []
    for combination in product(*params):
        experiment = dict(zip(param_names, combination))
        run = argparse.Namespace(**experiment)
        run = copy.deepcopy(run)  # Ensure a unique copy
        run = update_config(run)   
        experiments.append(run)
    return experiments

def run_experiment(run):
    """Run a single experiment configuration"""
    data_path, log_folder, use_logger, use_WB = run.data_path, run.log_folder, run.use_logger, run.use_WB

    if use_logger:
        logger = ns.utils.FileLogger(base_folder=log_folder)
        if logger.exists_experiment(run.__dict__): return

    for seed in run.seed:
        run.seed_run_i = seed
        print(f"\nSeed {seed} in {run.seed}")
        
        log_filename_tmp = os.path.join(log_folder, f'_tmp_log-{run.run_signature}-{logger.date}-seed_{seed}.csv') if use_logger else None
        
        if use_logger and logger.exists_run(run.run_signature, seed): 
            continue

        metrics = main_train(data_path, log_filename_tmp, use_WB, run)
        
        if use_logger:
            # logged_data = {**copy.deepcopy(run).__dict__}
            dicts_metrics = {
                'train': metrics[0], 'valid': metrics[1], 
                'test': metrics[2], 'training_info': metrics[3]}
            task_mrr = np.round(metrics[2]['task_mrr'], 3)
            logger.log(log_filename_tmp,run.__dict__,dicts_metrics)
            final_log = os.path.join(log_folder, 'indiv_runs', 
                f'_ind_log-{run.run_signature}-{logger.date}-{task_mrr}-seed_{seed}.csv')
            logger.finalize_log_file(log_filename_tmp,final_log)

    if use_logger: 
        logger.get_avg_results(run.__dict__, run.run_signature, run.seed)

def main():
    config = ExperimentConfig()
    # setup_tf()
    wandb.login()
    experiments = generate_experiments(config)

    print(f"Running {len(experiments)} experiments:")
    for idx, run in enumerate(experiments):
        print(f"\nExperiment {idx+1}/{len(experiments)}")

        if not os.path.exists(os.path.join(run.data_path, run.dataset_name)):
            print('skipping, dataset not existing', os.path.join(run.data_path, run.dataset_name))
        if run.model_name == 'no_reasoner' and run.grounder != 'backward_0_1':
            print('skipping, selec grounder 0_1 with no reasoner', run.grounder)
        assert os.path.exists(os.path.join(run.data_path, run.dataset_name, run.rules_file)), 'Rules file not found'

        run_experiment(run)

if __name__ == '__main__':
    main()
