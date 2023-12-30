import copy
import datetime
import os
from itertools import product
from train import main
import shutil as sh
import keras_ns as ns
from keras_ns.utils import MMapModelCheckpoint, NSParser

NUM_CPUS :int = 1  # set to a larger num to enable parallel processing


if __name__ == '__main__':

    base_path :str = "data"
    parallel :bool = False

    epochs: int = 120
    assert epochs > 0

    id_experiment :str = "000"
    output_folder :str = "results_" + id_experiment
    log_folder :str = output_folder
    src_folder :str = os.path.join(output_folder, "src")
    dataset_name :str = 'countries'
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    if not os.path.exists(log_folder): os.mkdir(log_folder)
    if not os.path.exists(src_folder): os.mkdir(src_folder)
    for filename in os.listdir("."):
        filepath = os.path.join(".", filename)
        if os.path.isfile(filepath):
            sh.copy2(filepath, src_folder)

    SEED = [0, 1, 2, 3, 4]
    E = [5, 10, 20, 50, 100, 200]
    DROPOUT = [0.]

    NEG_PER_SIDE = [1]
    R = [0.0]
    RR = [0.0]
    LR = [0.01]
    NUM_RULES = [1]
    HARD = [False]
    DEPTH = [0]
    VALID_SIZE = [None]
    KGE = ['distmult']  # ["distmult", "transe","complex", "rotate"]
    RULE_FILE = ["rules.txt"]  # rules
    all_args = []

    for seed, dropout, r, neg, e, lr, nr, h, dp, v, kge, rr in product(
            SEED, DROPOUT, R,
            NEG_PER_SIDE, E, LR,
            NUM_RULES, HARD, DEPTH,
            VALID_SIZE, KGE, RR):

        run_vars = (seed, dropout, r, neg, e, lr, nr, h, dp, v, kge, rr)

        # Base parameters
        parser = NSParser()
        args = parser.parse_args()
        args.run_signature = '_'.join(f'{v}' for v in run_vars)
        args.reasoner = "r2n"  # "latent_worlds"
        args.adaptation_layer = "identity"  # "dense", "sigmoid","identity"
        args.output_layer = "dense" # "wmc" or "kge" or "positive_dense" or "max"
        args.learning_rate = lr
        args.ragged = True
        args.num_rules = nr
        args.relation_entity_grounder_max_elements = 20
        args.debug = True
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
        args.train_file = "train_S2.txt"
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

        args.model = None
        args.batch_size = 10000
        # Full batch only for explain.
        args.eval_batch_size = -1
        args.test_batch_size = -1
        args.valid_size = v
        args.valid_negatives = 10
        args.valid_frequency = 1000
        args.engine_num_negatives = 0

        args.constant_embedding_size = (
            2 * args.kge_atom_embedding_size
            if args.kge == "complex" or args.kge == "rotate"
            else args.kge_atom_embedding_size)
        args.kge_regularization = r

        args.resnet_rule = False
        args.reasoner_depth = dp if nr > 0 else 0
        args.enabled_reasoner_depth = args.reasoner_depth
        args.reasoner_regularization_factor = rr
        args.reasoner_formula_hidden_embedding_size = args.kge_atom_embedding_size
        args.reasoner_dropout_rate = dropout
        args.reasoner_atom_embedding_size = args.kge_atom_embedding_size
        args.create_flat_rule_list = True

        all_args.append(args)


    def main_wrapper(args):
        logger = ns.utils.FileLogger(log_folder)
        if logger.exists(args.__dict__):
            print("Skipping", args)
            return
        else:
            time = str(datetime.datetime.now()).replace(":","-")
            log_filename_tmp = os.path.join(log_folder, '_tmp_log%s.csv' % time)
            log_filename = os.path.join(
                log_folder, 'log%s_%s.csv' % (args.run_signature, time))

            best_val, _, valid_acc, test_acc, _ = main(
                base_path,
                None,
                None,
                log_filename_tmp,
                args)

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

    for args in all_args:
        main_wrapper(args)
