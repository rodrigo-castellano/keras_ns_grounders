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
import ast
NUM_CPUS :int = 1  # set to a larger num to enable parallel processing

if __name__ == '__main__':

    base_path :str = "data"
    parallel :bool = False

    epochs: int = 10
    assert epochs > 0
    DATASET_NAME =  ['test_dataset'] #['kinship_family'] # ['pharmkg_supersmall','countries_s1','countries_s2','countries_s3'] # ['kinship_family'] #['countries_s1','countries_s2','countries_s3','pharmkg_supersmall','nations','kinship_family_small'] 
    MODIFIED_DATASET = [False]
    GROUNDER = ['backward_1'] #['backward_1','backward_prune_2','backward_2','backward_prune_3','backward_3',]  #['backward_prune_1','backward_1','backward_prune_2','backward_2','backward_prune_3','backward_3','domainbody','full']  
    KGE = ['complex']  # ["distmult", "transe","complex", "rotate"]
    MODEL_NAME =  ['dcr']#['no_reasoner','sbr','rnm','dcr','r2n']  
    RULE_MINER = ['amie','None'] 
    E = [100] 
    DEPTH = [1]
    SEED = [[0,1]]
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

    for dataset_name,modified_dataset, grounder, kge, model_name, rule_miner, e, dp, seed, neg, w_loss,  dropout, r, lr, nr, h,  v, rr in product(
            DATASET_NAME,MODIFIED_DATASET, GROUNDER, KGE, MODEL_NAME, RULE_MINER, E, DEPTH, SEED, NEG_PER_SIDE, WEIGHT_LOSS, DROPOUT, R,
            LR, NUM_RULES, HARD,  VALID_SIZE, RR ):  
    

        # Base parameters
        parser = NSParser()
        args = parser.parse_args()


        if modified_dataset:
            if 'backward' not in grounder:
                continue
            elif 'backward_prune_3'== grounder or 'backward_3' == grounder:
                continue
            else:
                pruning= 'p' if 'prune' in grounder else 'np'
                level = grounder[-1]
                dataset_name = dataset_name+'_reason_2'+pruning
                dataset_name = dataset_name
        if not os.path.exists(os.path.join(base_path, dataset_name)):
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

        if ('kinship_family_small_reason' in dataset_name) and model_name=='dcr' :
            print('skipping, dcr not valid for',dataset_name, run_vars)    
            continue

        if ('kinship_family_small' in dataset_name) and model_name=='r2n' and (grounder == 'backward_2' or grounder == 'backward_prune_2' or grounder == 'backward_3' or grounder == 'backward_prune_3'):
            print('skipping, r2n not valid for',dataset_name, run_vars)    
            continue

        if dataset_name == 'kinship_family_small' and grounder == 'backward_prune_3' and model_name == 'sbr':
            print('skipping, sbr doesnt work for kinship backward_3', run_vars)
            continue

        if (dataset_name == 'kinship_family' or dataset_name == 'pharmkg_full') and (model_name == 'sbr' or model_name=='r2n'):
            print('skipping,',model_name,' doesnt work for',dataset_name, run_vars)
            continue

        if (dataset_name == 'pharmkg_full' or dataset_name == 'kinship_family') and grounder != 'backward_1' and model_name=='no_reasoner':
            print('skipping, no_reasoner not needed if not with backw 1 for',dataset_name, run_vars)    
            continue

        elif 'nations' in dataset_name:
            if  (grounder == 'full'):
                print('skipping, grounder too heavy', run_vars)
                continue

        elif  ('pharm' in dataset_name):
            if  ( grounder == 'full'):
                print('skipping, grounder too heavy', run_vars)
                continue
            
        elif ('kinship' in dataset_name):
            if  (grounder == 'full' or grounder == 'domainbody'):
                print('skipping, grounder too heavy', run_vars)
                continue

        args.dataset_name = dataset_name
        args.grounder = grounder
        args.kge = kge
        args.model_name = model_name 
        args.rule_miner = rule_miner 
        args.modified_dataset = modified_dataset
        args.seed = seed
        args.facts_file = 'facts.txt'
        args.train_file = 'train.txt'  
        args.valid_file = 'valid.txt'
        args.test_file = 'test.txt'
        args.domain_file = 'domain2constants.txt'
        args.rules_file = 'rules.txt'

        if rule_miner == 'amie':
            args.rules_file = 'rules_amie.txt'
        elif rule_miner == 'ncrl':
            args.rules_file = 'rules_ncrl.txt'
        elif rule_miner == 'None':
            args.rules_file = 'rules.txt'
        else: # raise an error if the rule miner is not recognized
            raise ValueError('Rule miner not recognized for ', dataset_name)
        if not os.path.exists(os.path.join(base_path, dataset_name, args.rules_file)):
            # print('skipping, rules not existing', run_vars)
            continue

        args.test_negatives = None  # all possible negatives
        if dataset_name == 'pharmkg_full' or dataset_name == 'kinship_family':
            args.test_negatives = 1000
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
        args.format = "functional"
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
        run_vars = (args.dataset_name,grounder, kge, model_name, rule_miner, modified_dataset, e, dp, seed, neg, w_loss, dropout)
        args.keys_signature = ['dataset_name','grounder', 'kge', 'model_name', 'rule_miner', 'modified_dataset', 'e', 'dp', 'seed', 'neg', 'w_loss', 'dropout']
        args.run_signature = '-'.join(f'{v}' for v in run_vars)     
        all_args.append(args)




        def get_avg_results(log_folder,run_signature,seeds):
            # For every file with a different seed, read the results and take the average
            all_files = os.listdir(os.path.join(log_folder,'indiv_runs'))
            # get the files that contain the run_signature
            run_files = [file for file in all_files if run_signature in file]
            # get the number of files
            n_files = len(run_files)
            if n_files != len(seeds):
                return
            # for every file, read all the lines and if the line starts with 'all_data', take the values and add them to the array
            info = {}
            for file in run_files:
                with open(os.path.join(log_folder,'indiv_runs',file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith('All data'):
                            d = line.split(';')[1:]
                            info_exp = {el.split(':')[0] : ast.literal_eval(el.split(':')[1]) for el in d if el.split(':')[0] in ['train_acc', 'valid_acc', 'test_acc','time_run']}
                            # get the names of the metrics from the element 'metrics'
                            metrics = [ast.literal_eval(el.split(':')[1]) for el in d if el.split(':')[0] == 'metrics'][0]
                            print('metrics',metrics)
                            # append the key,values to the dictionary
                            for key in info_exp.keys():
                                if key in info:
                                    info[key].append(np.round(info_exp[key],3))
                                else:
                                    info[key] = [np.round(info_exp[key],3)]
            # for every key in the dictionary, take the average and the std
            for key in info.keys():
                avg = np.mean(info[key],axis=0)
                std = np.std(info[key],axis=0)
                info[key] = [list(avg),list(std)]
            return info,metrics
        
        def write_avg_results(args,filename,info_metrics,metrics_name):
            if 'contrastive_loss' in metrics_name: # delete it from the list metrics_name
                metrics_name.remove('contrastive_loss')
            # join the args.__dict__ with the info_metrics
            metrics = [str(element)+'_'+str(metric) for element in ['train','val','test'] for metric in list(metrics_name)]
            combined_names = ';'.join(list(args.__dict__.keys()) + [str(metric) for metric in metrics] )
            values_args = [str(v) for k,v in args.__dict__.items()] 
            values_metrics = []
            for k,v in info_metrics.items():
                for i in range(len(v[0])):
                    values_metrics.append(str([ np.round(v[0][i],3) , np.round(v[1][i],3) ]))
            combined_results = ';'.join(values_args + values_metrics)

            # Create a file for my results 
            hparam_folder = './hparamsearch/'
            if not os.path.exists(hparam_folder): os.mkdir(hparam_folder)
            print("Writing results to", filename)   
        
            # Check if the file hparam_folder+'headers.txt' exists, otherwise create it
            if not os.path.exists(hparam_folder+'headers.txt'):
                with open(hparam_folder+'headers.txt', 'w') as f:
                    f.write('sep=;\n')
                    f.write(combined_names)
            else: 
                with open(hparam_folder+'headers.txt', 'r') as f:
                    lines = f.readlines()
                    if combined_names not in lines:
                        with open(hparam_folder+'headers.txt', 'a') as f:
                            f.write('\n')
                            f.write(combined_names)

            with open(filename, 'a') as f: 
                empty = os.stat(filename).st_size == 0
                if empty:
                    f.write('sep=;\n')
                    f.write(combined_names)
                f.write('\n')
                f.write(combined_results)

    def main_wrapper(args): 
        # HPARAM SEARCH
        # Check if the experiment has already been run.
        # create a string for the run_vars, each substring separated by a '_'
        print("\nRun vars:", args.run_signature+'\n')

        # hparam_folder = './hparamsearch/'
        # if not os.path.exists(hparam_folder): os.mkdir(hparam_folder)
        # hparam_filename = hparam_folder+'hparamsearch.txt'
        # # If the file does not exist, create it, but do not write anything
        # if not os.path.exists(hparam_filename):
        #     with open(hparam_filename, 'w') as f:
        #         pass
        # # If the file exists, check if the run_vars are already in the file, if not, write them
        # else:
        #     with open(hparam_filename, 'r') as f:
        #         lines = f.readlines() 
        #         # print("Lines in file:\n", lines)
        #         # print("Run vars:\n", args.run_signature+'\n')
        #         if args.run_signature+'\n' in lines:
        #             print("Run vars already in file")
        #             return           
        
        # LOGGER
        # Results for every epoch will be saved in a folder 
        log_folder :str = "results"
        # Check if the logger exists, if so, skip the experiment, otherwise run it. Logger exists if all the arguments inside each file in the folder are the same as the current args
        logger = ns.utils.FileLogger(log_folder)
        if logger.exists(args.__dict__):
            print("\n\nSkipping training, it has been already done for", args.run_signature, "\n")
            return
        else:
            date = logger.get_date()
            # one folder is for temporal files, the other for the final log, and the other is when several runs are done, is like also a temporal folder
            log_filename = os.path.join(log_folder, 'log{}_{}.csv'.format(args.run_signature, date))
            os.makedirs(os.path.dirname(os.path.join(log_folder,'indiv_runs')), exist_ok=True)

        # try:
        n_seeds = len(args.seed)
        for i,seed in enumerate(args.seed):
            start = time.time()
            args.seed_run_i = seed
            log_filename_tmp = os.path.join(log_folder,'_tmp_log-{}-{}-seed_{}.csv'.format(args.run_signature,date,seed))
            

            # Check if the training has already been done for this seed
            # in log_filename_tmp, take up to the last two elements split by ';' to not take into account the time
            sub_signature = log_filename_tmp.split('-')[1:-2]
            # addd the seed
            sub_signature.append(str('seed_'+str(seed)))
            # read all the files
            all_files = os.listdir(os.path.join(log_folder,'indiv_runs'))
            found = False
            for file in all_files:
                # if the file contains the sub_signature, then the training has been done
                if all(sub in file for sub in sub_signature):
                    found = True
                    break
            if found:   
                print("Seed number ", seed, " in ", args.seed,'already done')
                continue

            
            print("Seed number ", seed, " in ", args.seed)
            # write in the tmp file 'sep=;' to separate the columns with a semicolon
            with open(log_filename_tmp, 'w') as f:
                f.write('sep=;\n')
            best_val, _, valid_acc, test_acc, _, train_acc, training_info = main(
                base_path,
                None,
                None,
                log_filename_tmp,
                args)

            end = time.time()
            time_run = end - start
            # The reuslts of the training have been written to tmp. write them as an individual run
            logged_data = copy.deepcopy(args)
            logged_data.train_acc = train_acc
            logged_data.valid_acc = valid_acc
            logged_data.best_val = best_val
            logged_data.test_acc = test_acc
            logged_data.metrics = list(training_info.keys())
            logged_data.time = time_run

            # write the info about the results in the tmp file 
            logger.log(logged_data.__dict__, log_filename_tmp)
            # Rename to not be temporal anymore
            log_filename_run = os.path.join(log_folder,'indiv_runs', '_ind_log-{}-{}-seed_{}.csv'.format(args.run_signature,date, i))
            if os.path.exists(log_filename_run):
                os.remove(log_filename_run)
            os.rename(log_filename_tmp, log_filename_run)
  
        # write the average results if we need to average over experiments
        if len(args.seed) > 1:
            info_metrics,metrics_name = get_avg_results(log_folder,args.run_signature,args.seed)
            write_avg_results(args,'./experiments/experiments.csv',info_metrics,metrics_name)

                
    for l,args in enumerate(all_args):
        print('Experiment',l,':',args.run_signature)
    for args in all_args:
        print('Experiment number ', all_args.index(args), ' out of ', len(all_args), ' experiments.')
        main_wrapper(args)
