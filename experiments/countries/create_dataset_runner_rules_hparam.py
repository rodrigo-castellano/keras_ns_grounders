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
from create_dataset_train_hparam import main 
import shutil as sh
import keras_ns as ns
from keras_ns.utils import MMapModelCheckpoint, NSParser
import time
import numpy as np

NUM_CPUS :int = 1  # set to a larger num to enable parallel processing

if __name__ == '__main__':

    base_path :str = "data"
    parallel :bool = False 

    RUNS_PER_CONFIG = [5]
    epochs: int = 100
    assert epochs > 0
    DATASET_NAME = ['countries_s1','countries_s2','countries_s3','pharmkg_supersmall']#['kinship_family_small','pharmkg_full','kinship_family_small'] #,'countries_s1','countries_s2','countries_s3','pharmkg_supersmall','nations','kinship_family_small'] #['kinship_family'] #['countries_s1','countries_s2','countries_s3','pharmkg_supersmall','nations','kinship_family_small'] 
    GROUNDER = ['backward_prune_2','backward_2','backward_prune_3','backward_3']#,'backward_prune_2','backward_3','backward_prune_3'] #['backward_2','backward_3','backward_step_2','backward_step_3'] #,'backward_1','backward_2','backward_3','domainbody','full','known',] # ['known','backward_1','backward_2','backward_3','domainbody','full'] 
    KGE = ['complex']  # ["distmult", "transe","complex", "rotate"]
    MODEL_NAME = ['dcr'] #,'sbr','rnm','dcr','r2n','no_reasoner',]# ['rnm','dcr','r2n','sbr','gsbr','cdcr','no_reasoner']  'gsbr' 'cdcr' not published yet
    RULE_MINER = ['amie','None'] #['amie','ncrl'] 
    E = [100] 
    DEPTH = [1]
    SEED = [[0]]
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

     

    def main_wrapper(args): 
        date = str(datetime.datetime.now()).replace(":","-")
        date = str(datetime.datetime.now()).replace(":","-").replace(" ","-")
        date = date[:date.index('.')]
        log_folder :str = "results"
        log_filename_tmp = os.path.join(log_folder, '_tmp_log_{}_{}.csv'.format(date,args.run_signature))

        main(base_path,None,None,log_filename_tmp,args)


    for args in all_args:
        print('Experiment number ', all_args.index(args), ' out of ', len(all_args), ' experiments.')
        main_wrapper(args)
