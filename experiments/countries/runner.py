import sys
sys.path.append('C:\\Users\\rodri\\Downloads\\PhD\\Review_grounders\\keras_ns_grounders')
sys.path.append('/home/castellanoontiv/keras_ns_grounders')
sys.path.append('/media/users/castellanoontiv/keras_ns_grounders/')
import os
import copy
import os
from itertools import product
from train import main
import shutil as sh
import keras_ns as ns
from keras_ns.utils import NSParser
import time
import tracemalloc


if __name__ == '__main__':

    base_path :str = "data"
    parallel :bool = False

    epochs: int = 10
    assert epochs > 0
    DATASET_NAME =  ['countries_s2'] #['kinship_family'] #['countries_s1','countries_s2','countries_s3','pharmkg_supersmall','nations','kinship_family_small'] 
    MODIFIED_DATASET = [False]
    GROUNDER = ['backward_1']  #['backward_1','backward_2','backward_3','domainbody','full']  
    KGE = ['complex']  # ["distmult", "transe","complex", "rotate"]
    MODEL_NAME =  ['dcr'] #['no_reasoner','sbr','rnm','dcr','r2n']  
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
    VALID_SIZE = [None]


    all_args = []

    for dataset_name,modified_dataset, grounder, kge, model_name, rule_miner, e, dp, seed, neg, w_loss,  dropout, r, lr, nr, v, rr in product(
            DATASET_NAME,MODIFIED_DATASET, GROUNDER, KGE, MODEL_NAME, RULE_MINER, E, DEPTH, SEED, NEG_PER_SIDE, WEIGHT_LOSS, DROPOUT, R,
            LR, NUM_RULES, VALID_SIZE, RR ):  
    

        # Base parameters
        parser = NSParser()
        args = parser.parse_args()


        if modified_dataset:
            if 'backward' not in grounder:
                continue
            else:
                level = grounder[-1]
                dataset_name = dataset_name+'_reason_'+level
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
        args.rule_weight = "number" # "embedding"
        args.semiring = "product"
        args.dropout_rate_embedder = dropout
        args.format = "functional"
        args.num_negatives = neg
        # DCR/R2N params
        args.signed = True
        args.temperature = 0.0
        args.aggregation_type = "max"
        args.filter_num_heads = 3
        args.filter_activity_regularization = 0.0
        args.epochs = epochs
        args.weight_loss = w_loss
        args.batch_size = -1
        # Full batch only for explain.
        args.val_batch_size = -1
        args.test_batch_size = 128
        args.cdcr_use_positional_embeddings = False
        args.cdcr_num_formulas = 3
        args.valid_size = v
        args.valid_negatives = 200
        args.valid_frequency = 3
        args.engine_num_negatives = 0
        args.engine_num_adaptive_constants = 0
        args.constant_embedding_size = (
            2 * args.kge_atom_embedding_size
            if args.kge == "complex" or args.kge == "rotate"
            else args.kge_atom_embedding_size)
        args.kge_regularization = r
        args.resnet = True
        args.reasoner_depth = dp if nr > 0 else 0
        args.enabled_reasoner_depth = args.reasoner_depth
        args.reasoner_regularization_factor = rr
        args.reasoner_formula_hidden_embedding_size = args.kge_atom_embedding_size
        args.reasoner_dropout_rate = dropout
        args.reasoner_atom_embedding_size = args.kge_atom_embedding_size
        args.create_flat_rule_list = True
        run_vars = (args.dataset_name,grounder, kge, model_name, rule_miner, modified_dataset, seed, neg,e)
        args.keys_signature = ['dataset_name','grounder', 'kge', 'model_name', 'rule_miner', 'modified_dataset', 'seed', 'neg','e']
        args.run_signature = '-'.join(f'{v}' for v in run_vars)     
        all_args.append(args)


 


    def main_wrapper(args): 

        print("\nRun vars:", args.run_signature+'\n')
        # LOGGER
        # Results for every epoch will be saved in a folder 
        log_folder :str = "results"
        log_folder_run = os.path.join(log_folder,'indiv_runs')
        log_folder_experiments = os.path.join(log_folder,'experiments')
        # Check if the logger exists, if so, skip the experiment, otherwise run it. Logger exists if all the arguments inside each file in the folder are the same as the current args
        logger = ns.utils.FileLogger(log_folder,log_folder_experiments,log_folder_run)
        if logger.exists_experiment(args.__dict__):
            print("Skipping training, it has been already done for", args.run_signature, "\n")
            # return

        date = logger.get_date()
        # try:
        n_seeds = len(args.seed)
        for i,seed in enumerate(args.seed):
            start = time.time()
            args.seed_run_i = seed
            log_filename_tmp = os.path.join(log_folder,'_tmp_log-{}-{}-seed_{}.csv'.format(args.run_signature,date,seed))
            if logger.exists_run(args.__dict__,log_filename_tmp,seed):   
                print("Seed number ", seed, " in ", args.seed,'already done')
                # continue

            print("Seed number ", seed, " in ", args.seed)
            # write in the tmp file 'sep=;' to separate the columns with a semicolon
            # tracemalloc.start()
            # snapshot1 = tracemalloc.take_snapshot()
            with open(log_filename_tmp, 'w') as f:
                f.write('sep=;\n')
            try:
                train_acc,valid_acc, test_acc,training_info = main(base_path,None,None,log_filename_tmp,args)
            except Exception as e:
                print('Error in experiment', args.run_signature, 'seed', seed, 'error:', e, '. Try again!')
                train_acc,valid_acc, test_acc,training_info = main(base_path,None,None,log_filename_tmp,args)
            # snapshot2 = tracemalloc.take_snapshot()
            # tracemalloc.stop()
            # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            # print("[ Top 10 differences ]")
            # for stat in top_stats[:20]:
            #     print(stat)
            end = time.time()
            time_run = end - start
            # The reuslts of the training have been written to tmp. write them as an individual run
            logged_data = copy.deepcopy(args)
            logged_data.train_acc = train_acc
            logged_data.valid_acc = valid_acc
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
            info_metrics,metrics_name = logger.get_avg_results(args.run_signature,args.seed)
            logger.write_avg_results(args.__dict__,log_folder_experiments,info_metrics,metrics_name)

                
    for l,args in enumerate(all_args):
        print('Experiment',l,':',args.run_signature)
    for args in all_args:
        print('Experiment number ', all_args.index(args), ' out of ', len(all_args), ' experiments.')
        main_wrapper(args)
