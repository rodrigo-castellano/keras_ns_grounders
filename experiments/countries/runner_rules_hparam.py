import sys
sys.path.append('C:\\Users\\rodri\\Downloads\\PhD\\Review_grounders\\code\\keras-ns')
sys.path.append('C:\\Users\\rodri\\Downloads\\PhD\\Review_grounders\\code\\keras-ns\\experiments\\countries')
sys.path.append('/home/castellanoontiv/keras_ns_grounders/experiments/countries')
sys.path.append('/home/castellanoontiv/keras_ns_grounders')
for p in sys.path:
    print(p)
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


# show the current path

NUM_CPUS :int = 1  # set to a larger num to enable parallel processing

if __name__ == '__main__':

    # Results for the hparam search 
    save_hparam_results = True 

    # Results for every epoch will be saved in a folder named log_folder
    log_folder :str = "results_002"
    if not os.path.exists(log_folder): os.mkdir(log_folder)

    dataset_name :str = 'countries'
    base_path :str = "data"
    parallel :bool = False
    epochs: int = 120
    assert epochs > 0

    SEED = [[0,1,2,3,4]]
    E = [100] 
    DROPOUT = [0.0]
    NEG_PER_SIDE = [1]
    R = [0.0]
    RR = [0.0]
    LR = [0.01]
    NUM_RULES = [1] #####################check HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    HARD = [False]
    DEPTH = [1]
    VALID_SIZE = [None]
    KGE = ['complex','distmult','transe']  # ["distmult", "transe","complex", "rotate"]
    WEIGHT_LOSS = [.5] #,1,0.7,0.3,0 
    RULE_FILE = ["rules.txt"]  # rules
    TRAIN_FILE = ["train_S2_p.txt"]#,["train_S1_p.txt","train_S2_p.txt","train_S3_p.txt"] # "train_S1_p_no_neighbor.txt",
    GROUNDER = ["backward","known","domain"] # 'backward', 'known', 'full', 'domain'
    MODEL_NAME = ['no_reasoner','dcr','r2n','sbr','rnm','gsbr','cdcr']  #['no_reasoner','dcr','r2n','sbr','rnm','gsbr','cdcr']
    all_args = []

    for train_file, grounder, kge, e, w_loss, seed, dropout, r, neg, lr, nr, h, dp, v, rr, model_name in product(
            TRAIN_FILE, GROUNDER, KGE, E, WEIGHT_LOSS, SEED, DROPOUT, R,
            NEG_PER_SIDE, LR,
            NUM_RULES, HARD, DEPTH,
            VALID_SIZE, RR, MODEL_NAME ):  
         
        run_vars = (train_file[:train_file.index('.')],grounder, kge, e,w_loss, seed, dropout, r, 
                    neg, lr, nr, h, dp, v, rr,model_name)

        # Base parameters
        parser = NSParser()
        args = parser.parse_args()
        args.run_signature = '_'.join(f'{v}' for v in run_vars) 
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
        args.train_file = train_file
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
        args.model = None
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
        metrics = [str(element)+'_'+str(metric) for element in ['train','val','test'] for metric in list(training_info.keys())]

        # Combine all values into a single comma-separated string
        combined_names = ';'.join(
            ['Task', 'Grounder', 'KGE', 'EmbedSize', 'WeightLoss_Task','Reasoner_depth','Model_name','Time'] + 
            [str(metric) for metric in metrics]
            )  
        
        combined_results = ';'.join(
            [args.train_file[:train_file.index('.')], args.grounder, str(args.kge),
                str(args.kge_atom_embedding_size),str(args.weight_loss),str(args.reasoner_depth),args.model_name]+
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
                f.write(combined_names)
            f.write('\n')
            f.write(combined_results)

    def main_wrapper(args): 
        # Check if the logger exists, if so, skip the experiment, otherwise run it.
        logger = ns.utils.FileLogger(log_folder)
        if logger.exists(args.__dict__):
            print("Skipping", args)
            return
        else:
            date = str(datetime.datetime.now()).replace(":","-")
            log_filename_tmp = os.path.join(log_folder, '_tmp_log%s.csv' % date)
            log_filename = os.path.join(
                log_folder, 'log%s_%s.csv' % (args.run_signature, date))

            # Create a dict with the keys coming from a list
            keys = ['loss', 'concept_loss', 'task_loss', 'concept_mrr', 'concept_hits@1@1', 'concept_hits@3@3', 'concept_hits@5@5', 'concept_hits@10@10', 'task_mrr', 'task_hits@1@1', 'task_hits@3@3', 'task_hits@5@5', 'task_hits@10@10']
            # create a dict with the keys and empty lists as values
            runs = 5
            test_acc_avg = np.zeros((len(keys),runs))
            valid_acc_avg = np.zeros((len(keys),runs))
            train_acc_avg = np.zeros((len(keys),runs))
            time_arr = np.zeros((runs))
            for i in range(runs):
                print("Run number ", i, " out of ", runs)
                start = time.time()
                args.seed_run_i = args.seed[i]
                best_val, _, valid_acc, test_acc, _, train_acc, training_info = main(
                    base_path,
                    None,
                    None,
                    log_filename_tmp,
                    args)
                test_acc_avg[:,i] = np.array(test_acc)
                valid_acc_avg[:,i] = np.array(valid_acc)
                train_acc_avg[:,i] = np.array(train_acc)
                # print("Test acc:", test_acc)
                # print("Valid acc:", valid_acc)
                # print("Train acc:", train_acc)
                print('test acc avg', test_acc_avg)
                # print('valid acc avg', valid_acc_avg)
                print('train acc avg', train_acc_avg)
                end = time.time()
                time_arr[i] = end - start
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

            # Split the args used for trainig from the logged data.
            if hasattr(args, 'model'):
                delattr(args, 'model')
            logged_data = copy.deepcopy(args)
            # Add some extra info to log.
            logged_data.valid_acc = valid_acc
            logged_data.best_val = best_val
            logged_data.test_acc = test_acc
            logged_data.log_filename = log_filename
            # Log the data to its final location.
            logger.log(logged_data.__dict__, log_filename_tmp)
            if os.path.exists(log_filename):
                os.remove(log_filename)
            os.rename(log_filename_tmp, log_filename)

            if save_hparam_results: 
                save_results(args, train_acc, valid_acc, test_acc, training_info, total_time, total_time_std, train_std, valid_std, test_std)
                

    for args in all_args:
        print('Experiment number ', all_args.index(args), ' out of ', len(all_args), ' experiments.')
        main_wrapper(args)
