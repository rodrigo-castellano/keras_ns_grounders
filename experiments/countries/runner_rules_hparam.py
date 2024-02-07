import sys
sys.path.append('C:\\Users\\rodri\\Downloads\\PhD\\Review_grounders\\keras_ns_grounders')
sys.path.append('/home/castellanoontiv/keras_ns_grounders')
sys.path.append('/media/users/castellanoontiv/keras_ns_grounders/')
# get current directory
import os
directory = os.getcwd()
import copy
import datetime  
import os
from itertools import product
from train_hparam import main
import shutil as sh
import keras_ns as ns
from keras_ns.utils import MMapModelCheckpoint, NSParser
import time
import numpy as np

NUM_CPUS :int = 1  # set to a larger num to enable parallel processing

if __name__ == '__main__':

    base_path :str = "data"
    parallel :bool = False

    RUNS_PER_CONFIG = [1]
    epochs: int = 10
    assert epochs > 0
    DATASET_NAME = ['countries_s2','countries_s1','countries_s2','countries_s3','pharmkg_supersmall','nations','kinship_family_small'] #['kinship_family'] #['countries_s1','countries_s2','countries_s3','pharmkg_supersmall','nations','kinship_family_small'] 
    GROUNDER = ['backward_1','backward_2','backward_3','domainbody','full','known',] # ['known','backward_1','backward_2','backward_3','domainbody','full'] 
    KGE = ['complex']  # ["distmult", "transe","complex", "rotate"]
    MODEL_NAME = ['rnm','no_reasoner','sbr','rnm','dcr','r2n']# ['rnm','dcr','r2n','sbr','gsbr','cdcr','no_reasoner']  'gsbr' 'cdcr' not published yet
    RULE_MINER = ['amie','None'] #['amie','ncrl'] 
    E = [100] 
    DEPTH = [1]
    SEED = [[0,1,2,3,4]]
    NEG_PER_SIDE = [1]
    WEIGHT_LOSS = [.5]  
    DROPOUT = [0.0]
    R = [0.0]
    RR = [0.0]
    LR = [0.01]
    NUM_RULES = [1] 
    HARD = [False]
    VALID_SIZE = [None]


    all_args = []

    for dataset_name, grounder, kge, model_name, rule_miner, e, dp, seed, neg, w_loss,  dropout, r, lr, nr, h,  v, rr, runs in product(
            DATASET_NAME, GROUNDER, KGE, MODEL_NAME, RULE_MINER, E, DEPTH, SEED, NEG_PER_SIDE, WEIGHT_LOSS, DROPOUT, R,
            LR, NUM_RULES, HARD,  VALID_SIZE, RR, RUNS_PER_CONFIG ):  
        
        run_vars = (dataset_name,grounder, kge, model_name, rule_miner, e, dp, seed, neg, w_loss, dropout)
        # Base parameters
        parser = NSParser()
        args = parser.parse_args()
        args.runs = runs
        args.run_signature = '_'.join(f'{v}' for v in run_vars) 

        args.train_file = 'train.txt'  
        args.valid_file = 'None.txt'
        args.test_file = 'test.txt'
        args.facts_file = 'facts.txt'
        args.domain_file = 'domain2constants.txt'
        args.rules_file = 'rules.txt'
        args.rule_miner = rule_miner

        if rule_miner == 'amie':
            args.rules_file = 'rules_amie.txt'
        elif rule_miner == 'ncrl':
            args.rules_file = 'rules_ncrl.txt'
        elif rule_miner == 'None':
            args.rules_file = 'rules.txt'
        else: # raise an error if the rule miner is not recognized
            raise ValueError('Rule miner not recognized for ', dataset_name)
        if not os.path.exists(os.path.join(base_path, dataset_name, args.rules_file)):
            print('skipping, rules not existing', run_vars)
            continue

        if 'countries' in dataset_name:
            # task is the last two letters of the dataset name
            task = dataset_name[-2:]
            if task == 's2' and (grounder == 'full'): # or grounder == 'domainbody'): domainbody sometimes gives problems
                print('skipping, grounder too heavy', run_vars)
                continue
            elif task == 's3' and (grounder == 'full' or grounder == 'domainbody'):
                print('skipping, grounder too heavy', run_vars)
                continue
        if dataset_name == 'kinship_family':
            if model_name == 'sbr':
                print('skipping, sbr doesnt work for kinship', run_vars)
                continue
        # elif 'nations' in dataset_name:
        #     if  (grounder == 'full'):
        #         print('skipping, grounder too heavy', run_vars)
        #         continue

        # elif  ('pharm' in dataset_name):
        #     if  ( grounder == 'full'):
        #         print('skipping, grounder too heavy', run_vars)
        #         continue
            
        # elif ('kinship' in dataset_name):
        #     if  (grounder == 'full' or grounder == 'domainbody'):
        #         print('skipping, grounder too heavy', run_vars)
        #         continue
        args.test_negatives = None  # all possible negatives
        if dataset_name == 'pharmkg_full' or dataset_name == 'kinship_family':
            args.test_negatives = 1000
        # args.reasoner = "r2n"  # "latent_worlds"
        args.adaptation_layer = "identity"  # "dense", "sigmoid","identity"
        args.output_layer = "dense" # "wmc" or "kge" or "positive_dense" or "max"
        args.learning_rate = lr
        args.ragged = True
        args.num_rules = 0 if model_name == "no_reasoner"  else nr
        args.relation_entity_grounder_max_elements = 20
        args.debug = False
        args.stop_gradient_on_kge_embeddings = False
        args.loss = "binary_crossentropy"
        args.kge_atom_embedding_size = e
        args.hard_rules = h
        args.rule_weight = "number" # "embedding"
        args.semiring = "product"
        args.dropout_rate_embedder = dropout
        args.seed = seed
        args.dataset_name = dataset_name
        args.format = "functional"
        args.grounder = grounder
        args.kge = kge
        args.num_negatives = neg
        # DCR/R2N params
        args.signed = True
        args.temperature = 3.0
        args.use_gumbel = True
        args.aggregation_type = "max"
        args.filter_num_heads = 1
        args.filter_activity_regularization = 0.0
        args.epochs = epochs 
        args.weight_loss = w_loss
        args.model_name = model_name  
        args.batch_size = -1
        # Full batch only for explain.
        args.eval_batch_size = -1
        args.test_batch_size = -1
        args.valid_size = v
        args.valid_negatives = 10 # 10
        args.valid_frequency = 1000
        args.engine_num_negatives = 0
        args.engine_num_adaptive_constants = 0  # HERE BEFOREEEEEE THERE WERE 3
        args.constant_embedding_size = (
            2 * args.kge_atom_embedding_size
            if args.kge == "complex" or args.kge == "rotate"
            else args.kge_atom_embedding_size)
        args.kge_regularization = r
        # args.resnet_rule = True
        args.resnet = True
        args.reasoner_depth = dp if nr > 0 else 0
        args.enabled_reasoner_depth = args.reasoner_depth
        args.reasoner_regularization_factor = rr
        args.reasoner_formula_hidden_embedding_size = args.kge_atom_embedding_size
        args.reasoner_dropout_rate = dropout
        args.reasoner_atom_embedding_size = args.kge_atom_embedding_size
        args.create_flat_rule_list = True

        all_args.append(args)

    def save_results(args, train_acc, valid_acc, test_acc, training_info, total_time, total_time_std, train_std, valid_std, test_std):
          
        # Select as metrics a list containing the keys of the dict training_info 
        # if the key constrastive_loss is in the dict, remove it
        if 'contrastive_loss' in training_info:
            del training_info['contrastive_loss']
        metrics = [str(element)+'_'+str(metric) for element in ['train','val','test'] for metric in list(training_info.keys())]

        # Combine all values into a single comma-separated string
        combined_names = ';'.join(
            ['Task', 'Grounder', 'KGE','Rule_Miner', 'EmbedSize', 'WeightLoss_Task','Reasoner_depth','Model_name','Time'] + 
            [str(metric) for metric in metrics]
            )  
        
        combined_results = ';'.join(
            [args.dataset_name, args.grounder, str(args.kge),args.rule_miner,str(args.kge_atom_embedding_size),
                str(args.weight_loss),str(args.reasoner_depth),args.model_name]+
            [str([total_time,total_time_std])] +
            [str([round(acc, 4), round(std, 4)]) for acc, std in zip(train_acc, train_std)] +
            [str([round(acc, 4), round(std, 4)]) for acc, std in zip(valid_acc, valid_std)] +
            [str([round(acc, 4), round(std, 4)]) for acc, std in zip(test_acc, test_std)]
            )  
        
        # Create a file for my results 
        date = str(datetime.datetime.now()).replace(":","-").replace(" ","_")
        date = date[:date.index('.')]
        hparam_folder = './hparamsearch/'
        if not os.path.exists(hparam_folder): os.mkdir(hparam_folder)
        results_filename = hparam_folder+'experiments.csv'  
        print("Writing results to", results_filename)   
            
        # Check if the file hparam_folder+'headers.txt' exists, otherwise create it
        if not os.path.exists(hparam_folder+'headers.txt'):
            with open(hparam_folder+'headers.txt', 'w') as f:
                f.write(combined_names)
                f.write('\n')

        with open(results_filename, 'a') as f: 
            empty = os.stat(results_filename).st_size == 0
            print("Empty file:", empty)
            if empty:
                f.write('\n')
                f.write(combined_names)
            f.write(combined_results)

    def main_wrapper(args): 
        # HPARAM SEARCH
        save_hparam_results = True
        # Check if the experiment has already been run.
        #create a string for the run_vars, each substring separated by a '_'
        print("\nRun vars:", args.run_signature+'\n')
        hparam_folder = './hparamsearch/'
        if not os.path.exists(hparam_folder): os.mkdir(hparam_folder)
        hparam_filename = hparam_folder+'hparamsearch.txt'
        # If the file does not exist, create it, but do not write anything
        if not os.path.exists(hparam_filename):
            with open(hparam_filename, 'w') as f:
                pass
        # If the file exists, check if the run_vars are already in the file, if not, write them
        else:
            with open(hparam_filename, 'r') as f:
                lines = f.readlines() 
                # print("Lines in file:\n", lines)
                # print("Run vars:\n", args.run_signature+'\n')
                if args.run_signature+'\n' in lines:
                    print("Run vars already in file")
                    return           
        
        # LOGGER
        # Results for every epoch will be saved in a folder named log_folder
        log_folder :str = "results"
        if not os.path.exists(log_folder): os.mkdir(log_folder)

        # Check if the logger exists, if so, skip the experiment, otherwise run it.
        logger = ns.utils.FileLogger(log_folder)
        if logger.exists(args.__dict__,signature=args.run_signature):
            print("\n\n\nSkipping training, it has been already done for", args.run_signature, "\n")
            return
        else:
            date = str(datetime.datetime.now()).replace(":","-")
            date = str(datetime.datetime.now()).replace(":","-").replace(" ","-")
            date = date[:date.index('.')]
            log_filename_tmp = os.path.join(log_folder, '_tmp_log_{}_{}.csv'.format(date,args.run_signature))
            log_filename = os.path.join(
                log_folder, 'log%s_%s.csv' % (args.run_signature, date))

        # try:
        for i in range(args.runs):
            print("Run number ", i, " out of ", runs)
            start = time.time()
            args.seed_run_i = args.seed[i]
            best_val, _, valid_acc, test_acc, _, train_acc, training_info = main(
                base_path,
                None,
                None,
                log_filename_tmp,
                args)
            if i == 0:
                # initialize the arrays with the number of keys in training_info
                keys = list(training_info.keys())
                test_acc_avg = np.zeros((len(keys),runs))
                valid_acc_avg = np.zeros((len(keys),runs))
                train_acc_avg = np.zeros((len(keys),runs))
                time_arr = np.zeros((runs))
            print('test_acc',test_acc)
            print('test acc avg', test_acc_avg) 
            print('train acc avg', train_acc_avg)
            test_acc_avg[:,i] = np.array(test_acc)
            valid_acc_avg[:,i] = np.array(valid_acc)
            train_acc_avg[:,i] = np.array(train_acc) 
            end = time.time()
            time_arr[i] = end - start
        # except Exception as e:
        #     print('Error in experiment', args.run_signature, e)
        #     return
        total_time =  np.mean(time_arr)
        total_time_std = np.std(time_arr)
        # Take the average across cols
        test_acc =  np.mean(test_acc_avg, axis=1)
        valid_acc = np.mean(valid_acc_avg, axis=1)
        train_acc = np.mean(train_acc_avg, axis=1)
        # Take the standard deviation across cols
        test_std = np.std(test_acc_avg, axis=1)
        valid_std = np.std(valid_acc_avg, axis=1)
        train_std = np.std(train_acc_avg, axis=1)


        # SAVE RESULTS FROM TRAINING IN LOG
        # # Split the args used for trainig from the logged data.
        if hasattr(args, 'seed_run_i'):
            delattr(args, 'seed_run_i')
        logged_data = copy.deepcopy(args)
        # # Add some extra info to log.
        logged_data.valid_acc = valid_acc
        logged_data.best_val = best_val
        logged_data.test_acc = test_acc
        logged_data.log_filename = log_filename
        # Log the data to its final location.
        logger.log(logged_data.__dict__, log_filename_tmp)
        if os.path.exists(log_filename):
            os.remove(log_filename)
        os.rename(log_filename_tmp, log_filename)


        # SAVE RESULTS FROM HPARAMSEARCH
        # Write the run_vars to the file
        with open(hparam_filename, 'a') as f:
            f.write('\n')
            f.write(args.run_signature)   
        if save_hparam_results: 
            save_results(args, train_acc, valid_acc, test_acc, training_info, total_time, total_time_std, train_std, valid_std, test_std)
                

    for args in all_args:
        print('Experiment number ', all_args.index(args), ' out of ', len(all_args), ' experiments.')
        main_wrapper(args)
