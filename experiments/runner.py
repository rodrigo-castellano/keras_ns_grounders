import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', 'ULTRA'))
sys.path.append(os.path.join(current_dir, '..', 'ns_lib'))
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy
from itertools import product
from train import main
import shutil as sh
import ns_lib as ns
from ns_lib.utils import NSParser
import numpy as np
import ast
import argparse
from model_utils import * 

import wandb 
wandb.login()

if __name__ == '__main__':

    import sys 
    # Define the file to redirect output to
    log_file = open('./experiments/training_logs.txt', 'a')
    sys.stdout = sys.stderr = log_file

    print("GPUs used: ", tf.config.experimental.list_physical_devices('GPU'))
    # tf.config.run_functions_eagerly(True)
    # Choose whether to save the results or not, and the folders where to save them
    use_logger = True
    use_WB = True
    log_folder :str = "./experiments/runs/"
    ckpt_folder :str = None #os.path.join(log_folder,'checkpoints')
    checkpoint_load = False
    base_path :str = "experiments/data"
    epochs: int = 80
    EARLY_STOPPING = True
    GLOBAL_SERIALIZATION = False
    LLM = False
    ULTRA = False
    ULTRA_WITH_KGE = False
    DATASET_NAME = ['countries_s1','countries_s2','countries_s3','nations','kinship_family','pharmkg_small','pharmkg_full','wn18rr']#,'FB15k237']
    GROUNDER = ['backward_1', 'backward_1_1','backward_2','backward_1_2','backward_3','backward_1_3']#,'domainbody','relationentity','full]
    KGE = ['complex']#,'rotate']  # ["distmult", "transe","complex", "rotate"]
    MODEL_NAME = ['no_reasoner','dcr','sbr','r2n']  
    RULE_MINER = ['amie','None'] 
    E = [100]#,300] 
    DEPTH = [1]
    SEED = [[0,1,2,3,4]]
    NEG_PER_SIDE = [2]
    WEIGHT_LOSS = [.5]  
    DROPOUT = [0.0]
    R = [0.0]
    RR = [0.0]
    LR = [0.01] #[0.01]
    LR_SCHEDULER = ['plateau'] #['plateau'] # None
    OPTIMIZER = ['adam'] #['adam'] #['None','adam']
    NUM_RULES = [1] 
    VALID_SIZE = [None]

    
    
    
    # This is in case I want to take inputs from the command line

    parser = argparse.ArgumentParser(description='Description of your script')  
    parser.add_argument("--d", default = None, help="dataset",nargs='+')
    parser.add_argument("--m", default = None, help="model",nargs='+')
    parser.add_argument("--g", default = None, help="grounder",nargs='+')
    parser.add_argument("--s", default = None, help="seed")

    args = parser.parse_args()
    if args.s is not None:
        SEED = [ast.literal_eval(args.s) ]
        assert isinstance(SEED, list) 
    if args.m is not None:
        MODEL_NAME = args.m
    if args.d is not None:
        DATASET_NAME = args.d
    if args.g is not None:
        GROUNDER = args.g

    del args.s
    del args.m
    del args.d  
    del args.g
    print('Running experiments for the following parameters:','DATASET_NAME:',DATASET_NAME,'GROUNDER:',GROUNDER,'MODEL_NAME:',MODEL_NAME,'SEED:',SEED)
    
    # Do the hparam search
    all_args = []
    for dataset_name,grounder, kge, model_name, rule_miner, e, dp, seed, neg, w_loss,  dropout, r, lr,lr_sched,optimizer, nr, rr in product(
            DATASET_NAME,GROUNDER, KGE, MODEL_NAME, RULE_MINER, E, DEPTH, SEED, NEG_PER_SIDE, WEIGHT_LOSS, DROPOUT, R,
            LR,LR_SCHEDULER,OPTIMIZER, NUM_RULES, RR ):  

        run_vars = (dataset_name,grounder, kge, model_name, rule_miner, neg, e)
        
        if not os.path.exists(os.path.join(base_path, dataset_name)):
            print('skipping, dataset not existing', run_vars)
            continue

        # Define sets for quick membership testing
        light_datasets_full = {'countries_s1', 'countries_s2', 'pharmkg_small'}
        heavy_datasets_domainbody_relationentity = {'countries_s3', 'wn18rr', 'pharmkg_full', 'FB15K', 'kinship_family'}

        # Discern the datasets for which the grounders full, domainbody, and relationentity are too heavy to run
        if (grounder == 'full' and dataset_name not in light_datasets_full) or \
        ((grounder in {'domainbody', 'relationentity'}) and dataset_name in heavy_datasets_domainbody_relationentity) or \
        (model_name == 'no_reasoner' and grounder not in {'backward_1'}):
            continue

        args.dataset_name = dataset_name
        args.grounder = grounder
        args.kge = kge
        args.model_name = model_name 
        args.rule_miner = rule_miner 
        args.seed = seed
        if (dataset_name == 'pharmkg_full' or dataset_name == 'wn18rr' or dataset_name == 'FB15k237'): # For heavy datasets, run only one seed
            args.seed = [0]
        args.kge_atom_embedding_size = e
        args.batch_size = 256 # Full batch only for explain.
        args.val_batch_size = 256
        args.test_batch_size = 128 if dataset_name in heavy_datasets_domainbody_relationentity else 256
        args.facts_file = 'facts.txt'
        args.train_file = 'train.txt'  
        args.valid_file = 'valid.txt'
        args.test_file = 'test.txt'
        args.domain_file = 'domain2constants.txt'
        args.rules_file = 'rules.txt' 

        # Select the rules file
        rules_files = {'amie': 'rules_amie.txt', 'ncrl': 'rules_ncrl.txt', 'None': 'rules.txt'}
        args.rules_file = rules_files.get(rule_miner) or ValueError(f'Rule miner not recognized for {dataset_name}')
        if not os.path.exists(os.path.join(base_path, dataset_name, args.rules_file)):
            continue

        # Data params
        args.corrupt_mode = 'TAIL' if ('countries' in dataset_name or 'wn18rr' in dataset_name) else 'HEAD_AND_TAIL'
        # args.corrupt_mode = 'TAIL' if ('countries' in dataset_name or dataset_name=='wn18rr' or dataset_name=='FB15k237' or dataset_name== 'pharmkg_full') else 'HEAD_AND_TAIL'
        args.num_negatives = neg  
        args.valid_negatives = 100  
        args.test_negatives = None  # all possible negatives
        args.ragged = True
        args.format = "functional"
        args.engine_num_negatives = 0
        args.engine_num_adaptive_constants = 0
        args.constant_embedding_size = (
            2 * args.kge_atom_embedding_size
            if args.kge == "complex" or args.kge == "rotate"
            else args.kge_atom_embedding_size)
        args.predicate_embedding_size = (
            2 * args.kge_atom_embedding_size
            if args.kge == 'complex'
            else args.kge_atom_embedding_size)
        # KGE params
        args.dropout_rate_embedder = dropout
        args.kge_regularization = r
        # Model params
        args.learning_rate = lr
        args.lr_sched = lr_sched
        args.optimizer = 'adam'
        args.early_stopping = EARLY_STOPPING
        args.epochs = epochs if not args.early_stopping else 1500
        args.num_rules = 0 if model_name == "no_reasoner"  else nr
        args.loss = "binary_crossentropy"
        args.weight_loss = w_loss
        args.cdcr_use_positional_embeddings = False
        args.cdcr_num_formulas = 3
        args.valid_frequency = 1
        args.resnet = True
        args.reasoner_depth = dp if nr > 0 else 0
        args.reasoner_regularization_factor = rr
        args.reasoner_formula_hidden_embedding_size = args.kge_atom_embedding_size
        args.reasoner_dropout_rate = dropout
        args.kge_dropout_rate = dropout
        args.reasoner_atom_embedding_size = args.kge_atom_embedding_size
        # DCR/R2N params
        args.signed = True
        args.temperature = 0.0
        args.aggregation_type = "max"
        args.filter_num_heads = 3
        args.filter_activity_regularization = 0.0
        # Other
        # args.adaptation_layer = "identity"  # "dense", "sigmoid","identity"
        # args.output_layer = "dense" # "wmc" or "kge" or "positive_dense" or "max"
        # args.relation_entity_grounder_max_elements = 20
        # args.semiring = "product"

        run_vars = (args.dataset_name,grounder, kge, model_name, rule_miner, neg, e)
        args.keys_signature = ['dataset_name','grounder', 'kge', 'model_name', 'rule_miner','neg','e',]
        args.run_signature = '-'.join(f'{v}' for v in run_vars)    

        args.device = 'cpu' # if not tf.config.experimental.list_physical_devices('GPU') else 'gpu'
        args.global_serialization = GLOBAL_SERIALIZATION
        args.use_ultra = ULTRA
        args.use_ultra_with_kge = ULTRA_WITH_KGE
        args.use_llm = LLM
        if args.use_ultra:
            args.run_signature = 'ultra-'+args.run_signature 
        elif args.use_ultra_with_kge:
            args.run_signature = 'ultra_kge-'+args.run_signature
        elif args.use_llm:
            args.run_signature = 'llm-'+args.run_signature
        
        args.ckpt_filepath = (os.path.join(ckpt_folder, args.run_signature,args.run_signature) if ckpt_folder else None) 
        args.checkpoint_load = os.path.join(ckpt_folder, args.run_signature,args.run_signature) if checkpoint_load else None
        args.kge_checkpoint_load = None #os.path.join(ckpt_folder, args.run_signature,,args.run_signature) if checkpoint_load else None 
        # append a hard copy of the args to the list of all_args
        all_args.append(copy.deepcopy(args)) 




    def main_wrapper(args,log_folder): 

        print("\nRun vars:", args.run_signature+'\n')

        # LOGGER (can skip if not used)

        # Check if the logger exists, if so, skip the experiment, otherwise run it 
        # Logger exists if, all the arguments inside each file in the folder runs, are the same as the current args
        if use_logger:
            logger = ns.utils.FileLogger(folder=log_folder,folder_experiments=os.path.join(log_folder,'experiments'),folder_run=os.path.join(log_folder,'indiv_runs'))
            if logger.exists_experiment(args.__dict__):
                print("Skipping training, it has been already done for", args.run_signature, "\n")
                #return

        for seed in args.seed:
            args.seed_run_i = seed

            if use_logger:
                date = logger.get_date()
                log_filename_tmp = os.path.join(log_folder,'_tmp_log-{}-{}-seed_{}.csv'.format(args.run_signature,date,seed))
                if logger.exists_run(args.__dict__,log_filename_tmp,seed):   
                    print("Seed number ", seed, " in ", args.seed,'already done')
                    continue
                # else:
                #     print("Seed number ", seed, " not done. Exit")
                #     continue
                with open(log_filename_tmp, 'w') as f:
                    f.write('sep=;\n')
            else:   
                log_filename_tmp = None

            print("Seed number ", seed, " in ", args.seed)
            train_acc,valid_acc, test_acc,training_info = main(base_path,None,log_filename_tmp,use_WB,args)


            if use_logger:
                # Rename the temporal file to the final file, and include the results in the logger
                logged_data = copy.deepcopy(args)
                logged_data.train_acc = train_acc
                logged_data.valid_acc = valid_acc
                logged_data.test_acc = test_acc
                logged_data.metrics = list(training_info.keys())
                logged_data.time_train = args.time_train
                logged_data.time_inference = args.time_inference
                logged_data.time_ground_train = args.time_ground_train
                logged_data.time_ground_valid = args.time_ground_valid
                logged_data.time_ground_test = args.time_ground_test
                # write the info about the results in the tmp file 
                logger.log(logged_data.__dict__, log_filename_tmp)
                # Rename to not be temporal anymore
                log_filename_run = os.path.join(log_folder,'indiv_runs', '_ind_log-{}-{}-{}-seed_{}.csv'.format(
                                                            args.run_signature,date,np.round(test_acc[-4],3),seed))
                if os.path.exists(log_filename_run):
                    os.remove(log_filename_run)
                os.rename(log_filename_tmp, log_filename_run)
                
        if use_logger:
            # write the average results if we need to average over experiments
            # if len(args.seed) > 1:
            info_results,metrics_name = logger.get_avg_results(args.run_signature,args.seed)
            if info_results is not None:
                logger.write_avg_results(args.__dict__,info_results,metrics_name)

                
    for args in all_args:
        print('Experiment number ', all_args.index(args), ' out of ', len(all_args), ' experiments.')
        main_wrapper(args,log_folder)