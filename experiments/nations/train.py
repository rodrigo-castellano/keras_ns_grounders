import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
#import tensorflow_ranking as tfr
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)

########to limit the numbers of CORE
# num_threads = 5
# os.environ["OMP_NUM_THREADS"] = "5"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "5"
# os.environ["TF_NUM_INTEROP_THREADS"] = "5"
#
# tf.config.threading.set_inter_op_parallelism_threads(
#     num_threads
# )
# tf.config.threading.set_intra_op_parallelism_threads(
#     num_threads
# )
# tf.config.set_soft_device_placement(True)


import argparse
import keras_ns as ns
from itertools import product
import numpy as np
from os.path import join
import random
import pickle

from dataset import KGCDataHandler, read_domains
from model import CollectiveModel
from keras.callbacks import CSVLogger
from keras_ns.logic.commons import Atom, Domain, Rule, RuleLoader
from keras_ns.nn.kge import KGEFactory
from keras_ns.utils import MMapModelCheckpoint, KgeLossFactory, read_file_as_lines

def get_arg(args, name: str, default=None, assert_defined=False):
    value = getattr(args, name) if hasattr(args, name) else default
    if assert_defined:
        assert value is not None, 'Arg %s is not defined: %s' % (name, str(args))
    return value

def main(base_path, output_filename, kge_output_filename, log_filename, args):

    # print("Num GPUs Available: ", len(gpus))
    csv_logger = CSVLogger(log_filename, append=True, separator=';')
    print('ARGS', args)

    seed = get_arg(args, 'seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Enable this to find to debug NANs.
    #tf.debugging.experimental.enable_dump_debug_info(
    #    "/tmp/tfdbg2_logdir",
    #    tensor_debug_mode="FULL_HEALTH",
    #    circular_buffer_size=-1)

    # Params
    ragged = get_arg(args, 'ragged', None, True)
    valid_frequency = get_arg(args, 'valid_frequency', None, True)

    # Data Loading
    data_handler = KGCDataHandler(
        dataset_name=args.dataset_name,
        base_path=base_path,
        format=get_arg(args, 'format', None, True),
        domain_file='domain2constants.txt',
        train_file=get_arg(args, 'train_file', None, "train_S1.txt"),
        valid_file="valid.txt",
        test_file="test.txt",
        fact_file="facts.txt")

    dataset_train = data_handler.get_dataset(
        split="train",
        number_negatives=args.num_negatives)
    #dataset_valid = data_handler.get_dataset(
    #    split="valid",
    #    number_negatives=args.valid_negatives, corrupt_mode='TAIL')
    dataset_test = data_handler.get_dataset(split="test", corrupt_mode='TAIL')
    dataset_test_positive_only = data_handler.get_dataset(
        split="test", number_negatives=0, corrupt_mode='TAIL')
    fol = data_handler.fol

    ### defining rules and grounding engine
    enable_rules = (args.reasoner_depth > 0 and args.num_rules > 0)
    if enable_rules:
        # For KGEs with no domains.
        # domains = {Rule.default_domain(): fol.domains[0]}

        # TODO: get domains from fol.
        _, domain2constants = read_domains(
            os.path.join(base_path, 'countries', 'domain2constants.txt'))
        domains = {"countries": Domain(name='countries',
                                       constants=domain2constants['countries']),
                   "subregions": Domain(name='subregions',
                                        constants=domain2constants['subregions']),
                   "regions": Domain(name='regions',
                                     constants=domain2constants['regions'])}
        rules = [
            Rule(name='f1',
                 var2domain={"X": "countries", "W": "subregions", "Z": "regions", "Y": "countries", "K": "countries"},
                 # body=["locatedIn(X,W)", "locatedIn(W,Z)", "neighborOf(Y,X)"],
                 # S3
                 #body=["neighborOf(X,Y)", "neighborOf(Y,K)", "locatedIn(K,Z)"],
                 # S2
                 body=["neighborOf(X,Y)", "locatedIn(Y,Z)", "locatedIn(X,W)", "locatedIn(W,Z)"],
                 # S1
                 #body=["locatedIn(X,W)", "locatedIn(W,Z)"],
                 head=["locatedIn(X,Z)"]),
        ]
        engine = ns.grounding.DomainBodyGrounder(domains=domains, rules=rules,
                                                 exclude_symmetric=True,
                                                 exclude_query=True)
    else:
        rules = []
        engine = None

    serializer = ns.serializer.LogicSerializer(
       predicates=fol.predicates, domains=fol.domains, debug=args.debug)

    # KGE
    kge_embedder = KGEFactory(args.kge)
    assert kge_embedder is not None

    # The model can be built here or passed from the outside in case of
    # usage of a pre-trained one.
    model = CollectiveModel(
        fol, rules,
        kge_embedder, args.kge_regularization,
        constant_embedding_size=args.constant_embedding_size,
        kge_atom_embedding_size=args.kge_atom_embedding_size,
        dropout_rate_embedder=args.dropout_rate_embedder,
        reasoner_single_model=get_arg(args, 'reasoner_single_model', False),
        reasoner_atom_embedding_size=args.reasoner_atom_embedding_size,
        reasoner_formula_hidden_embedding_size=args.reasoner_formula_hidden_embedding_size,
        reasoner_regularization=args.reasoner_regularization_factor,
        reasoner_dropout_rate=args.reasoner_dropout_rate,
        reasoner_depth=args.reasoner_depth,
        aggregation_type=args.aggregation_type,
        signed=args.signed,
        temperature=args.temperature,
        use_gumbel=args.use_gumbel,
        filter_num_heads=args.filter_num_heads,
        filter_activity_regularization=args.filter_activity_regularization)

    # Preparing data as generators for model fit
    data_gen_train = ns.dataset.DataGenerator(
        dataset_train, fol, serializer, engine,
        batch_size=args.batch_size, ragged=ragged)

    #data_gen_valid = ns.dataset.DataGenerator(
    #    dataset_valid, fol, serializer, engine,
    #    batch_size=args.eval_batch_size, ragged=ragged)

    data_gen_test = ns.dataset.DataGenerator(
        dataset_test, fol, serializer, engine,
        batch_size=args.eval_batch_size, ragged=ragged)

    data_gen_test_positive_only = ns.dataset.DataGenerator(
        dataset_test_positive_only, fol, serializer, engine,
        batch_size=args.eval_batch_size, ragged=ragged)

    _ = next(iter(data_gen_train))[0]

    #Loss
    loss_name = get_arg(args, 'loss', 'binary_crossentropy')
    loss = KgeLossFactory(loss_name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    metrics = [ns.utils.MRRMetric(),
               #ns.utils.HitsMetric(1),
               #ns.utils.HitsMetric(3),
               #ns.utils.HitsMetric(5),
               #ns.utils.HitsMetric(10)
               ]
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics,
                  run_eagerly=False)

    callbacks = []
    callbacks.append(csv_logger)
    # best_model_callback = MMapModelCheckpoint(model, "val_accuracy", frequency=valid_frequency)
    # callbacks.append(best_model_callback)

    model.fit(data_gen_train,
              epochs=args.epochs,
              callbacks=callbacks)
              #validation_data=data_gen_valid,
              #validation_freq=valid_frequency)
    # best_model_callback.restore_weights()

    if output_filename is not None:
        print('Saving model weights to', output_filename)
        model.save_weights(output_filename, overwrite=True)

    print("\nEvaluation", flush=True)
    valid_train = model.evaluate(data_gen_train)
    valid_accuracy = -1.0  # model.evaluate(data_gen_valid)
    test_accuracy = model.evaluate(data_gen_test)
    print('Results',
          'Train', valid_train,
          'Val', valid_accuracy,
          'Test', test_accuracy,
          flush=True)

    if enable_rules:
        # Switching to explain mode
        model.explain_mode()

        print('\nExplain With Negatives', flush=True)
        expls = model.predict(data_gen_test)
        for d, expl_layer_output in enumerate(expls):
            # [-1] gets the last reasoning depth output.
            rules = expl_layer_output[-1]
            for r, inputs in rules.items():
                print(model.reasoning[d].rules_embedders[r].explain(*inputs))

        print('\nExplain Positives', flush=True)
        expls = model.predict(data_gen_test_positive_only)
        for d, expl_layer_output in enumerate(expls):
            # [-1] gets the last reasoning depth output.
            rules = expl_layer_output[-1]
            for r, inputs in rules.items():
                print(model.reasoning[d].rules_embedders[r].explain(*inputs))

        # Switching to explain mode
        data_gen_test_explain = ns.dataset.DataGenerator(dataset_test, fol, serializer, engine,batch_size=-1, ragged=ragged)
        expls = model.predict(data_gen_test_explain)
        for d, expl_layer_output in enumerate(expls):
            # [-1] gets the last reasoning depth output.
            rules = expl_layer_output[-1]
            for r, inputs in rules.items():
                print(model.reasoning[d].rules_embedders[r].explain(*inputs))

    # return best_model_callback.best_val, 0,  valid_accuracy, test_accuracy, model
    return 0.0, 0.0, valid_accuracy, test_accuracy, model
