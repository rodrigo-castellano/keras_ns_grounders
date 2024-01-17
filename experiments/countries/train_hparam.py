import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from typing import Dict, List
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

import time
import argparse
import keras_ns as ns
from itertools import product
import numpy as np
from os.path import join
import random
import pickle

from dataset import KGCDataHandler, build_domains
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

def read_rules(path,args):
    print('Reading rules')
    rules = []
    #open file data/nations/rules.txt and real all the lines
    with open(path, 'r') as f:
        for line in f:
            # if len(rules) < 11:
            # split by :
            line = line.split(':')
            # first element is the name of the rule
            rule_name = line[0]
            # second element is the weight of the rule
            rule_weight = float(line[1].replace(',', '.'))
            # third element is the rule itself. Split by ->
            rule = line[2].split('->')
            # second element is the head of the rule
            rule_head = rule[1]
            # remove the \n from the head and the space
            rule_head = [rule_head[1:-1]]
            # first element is the body of the rule
            rule_body = rule[0]
            # split the body by ,
            rule_body = rule_body.split(', ')
            # for every body element, if the last character is a " ", remove it
            for i in range(len(rule_body)):
                if rule_body[i][-1] == " ":
                    rule_body[i] = rule_body[i][:-1]
            # Take the vars of the body and head and put them in a dictionary
            all_vars = rule_body + rule_head
            var_names = {}
            for i in range(len(all_vars)):
                # split the element of the body by (
                open_parenthesis = all_vars[i].split('(')
                # Split the second element by )
                variables = open_parenthesis[1].split(')')
                # divide the variables by ,
                variables = variables[0].split(',')
                # Create a dictionary with the variables as keys and the value "countries" as values
                if args.dataset_name == 'nations':
                    for var in variables:
                        var_names[var] = "countries"
                elif ('countries' in args.dataset_name) or ('test_dataset' in args.dataset_name):
                        var_names = {"X": "countries", "W": "subregions", "Z": "regions", "Y": "countries", "K": "countries"}
                elif 'kinship' in args.dataset_name:
                    var_names = {"x": "people", "y": "people", "z": "people"}
                    
            # print all the info
            print('number of rules: ', len(rules))
            if len(rules) < 1001:
                print('rule name: ', rule_name, 'rule weight: ', rule_weight, 'rule head: ', rule_head, 
                    'rule body: ', rule_body, 'var_names: ', var_names)
            rules.append(Rule(name=rule_name,var2domain=var_names,body=rule_body,head=rule_head))
    return rules

def main(base_path, output_filename, kge_output_filename, log_filename, args):

    csv_logger = CSVLogger(log_filename, append=True, separator=';')
    print('\nARGS', args,'\n')

    seed = get_arg(args, 'seed_run_i', 0)
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
        domain_file= args.domain_file,
        train_file= args.train_file,
        valid_file=args.valid_file,
        test_file= args.test_file,
        fact_file= args.facts_file)
    
    dataset_train = data_handler.get_dataset(
        split="train",
        number_negatives=args.num_negatives)
    # print('DATASET_TRAIN num queries', len(dataset_train),dataset_train)
    # print('DATASET_TRAIN query', dataset_train[0][0])
    # print('DATASET_TRAIN label', len(dataset_train[0][1]), dataset_train[0][1]) 
    # dataset_valid = data_handler.get_dataset(
    #    split="valid",number_negatives=args.valid_negatives, corrupt_mode='TAIL')
    dataset_test = data_handler.get_dataset(split="test", corrupt_mode='TAIL')
    # print('DATASET_test query', dataset_test[0][0])
    # print('DATASET_test label', dataset_test[0][1])
    # dataset_test_positive_only = data_handler.get_dataset(
        # split="test", number_negatives=0, corrupt_mode='TAIL')
    
    fol = data_handler.fol
    # print('FOL info\n', fol, flush=True)
    # print('FOL domains', fol.domains, '\n',' FOL predicates', fol.predicates, '\n', 'FOL name2domain', fol.name2domain, flush=True)
    domain2adaptive_constants: Dict[str, List[str]] = None
    num_adaptive_constants = get_arg(args, 'engine_num_adaptive_constants', 0)

    # Domains are used up to the serializer, the model assumes that all
    # constants are in teh same domain.

    ### defining rules and grounding engine
    enable_rules = (args.reasoner_depth > 0 and args.num_rules > 0)
    if enable_rules: 
        rules = read_rules(join(base_path, args.dataset_name, args.rules_file),args)
        # For KGEs with no domains.
        # domains = {Rule.default_domain(): fol.domains[0]}

        domain2adaptive_constants = {
            d.name : ['__adaptive_%s_%d' % (d.name, i)
                    for i in range(num_adaptive_constants)]
            for d in fol.domains
            }

        if args.grounder == 'full':
            engine = ns.grounding.PlaceholderGeneratorFullGrounder(
                domains={d.name:d for d in fol.domains},
                rules=rules,
                domain2adaptive_constants=domain2adaptive_constants,
                exclude_symmetric=True,
                exclude_query=False)
        elif args.grounder == 'domain':
            engine = ns.grounding.DomainFullGrounder(domains={d.name:d for d in fol.domains},
                                                    rules=rules,
                                                    exclude_symmetric=True,
                                                    exclude_query=False)
        elif args.grounder == 'known':
            engine = ns.grounding.KnownBodyGrounder(rules, facts=list(data_handler.train_known_facts_set))
        
        elif 'backward' in args.grounder:
            num_steps = int(args.grounder.split('_')[1])
            print('Using backward chaining with %d steps' % num_steps)
            engine = ns.grounding.BackwardChainingGrounder(rules, facts=list(data_handler.train_known_facts_set),
                                                        domains={d.name:d for d in fol.domains},
                                                        num_steps=num_steps)
        elif args.grounder == 'domainbody':
            engine = ns.grounding.DomainBodyGrounder(domains={d.name:d for d in fol.domains},
                                                    rules=rules,
                                                    exclude_symmetric=True,
                                                    exclude_query=False)
    else:
        rules = []
        engine = None

    serializer = ns.serializer.LogicSerializerFast(
        predicates=fol.predicates, domains=fol.domains,
        constant2domain_name=fol.constant2domain_name,
        domain2adaptive_constants=domain2adaptive_constants)

    # KGE
    print('Loading KGE', args.kge)
    kge_embedder = KGEFactory(args.kge)
    assert kge_embedder is not None

    # The model can be built here or passed from the outside in case of
    # usage of a pre-trained one.
    model = CollectiveModel(
        fol, rules,
        kge_embedder, args.kge_regularization,
        model_name=get_arg(args, 'model_name', 'dcr'),
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
        resnet= get_arg(args, 'resnet', False), 
        temperature=args.temperature,
        use_gumbel=args.use_gumbel,
        filter_num_heads=args.filter_num_heads,
        filter_activity_regularization=args.filter_activity_regularization,
        num_adaptive_constants=num_adaptive_constants)

    # Preparing data as generators for model fit
    print('Generating train data')
    start = time.time()
    data_gen_train = ns.dataset.DataGenerator(
        dataset_train, fol, serializer, engine,
        batch_size=args.batch_size, ragged=ragged)
    end = time.time()
    print("Time to create data generator train: ", np.round(end - start,2))
    # print('batch 0 from data_gen_train', data_gen_train[0][0])
    # print('batch 0 from data_gen_train',data_gen_train.__getitem__(0))
    # print("\n\n\n\n\n\n\n\nTime to create data generator train: ", np.round(end - start,2))
    # print(paco)
    # start = time.time()
    # data_gen_valid = ns.dataset.DataGenerator(
    #    dataset_valid, fol, serializer, engine,
    #    batch_size=args.eval_batch_size, ragged=ragged)
    # end = time.time()
    # print("Time to create data generator valid: ",  np.round(end - start,2))
    start = time.time()
    data_gen_test = ns.dataset.DataGenerator(
        dataset_test, fol, serializer, engine,
        batch_size=args.eval_batch_size, ragged=ragged)
    end = time.time()
    print("Time to create data generator test: ",  np.round(end - start,2))

    # data_gen_test_positive_only = ns.dataset.DataGenerator(
    #     dataset_test_positive_only, fol, serializer, engine,
    #     batch_size=args.eval_batch_size, ragged=ragged)

    #Loss
    loss_name = get_arg(args, 'loss', 'binary_crossentropy')
    loss = KgeLossFactory(loss_name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    metrics = [
                ns.utils.MRRMetric(),
                ns.utils.HitsMetric(n=1),
                ns.utils.HitsMetric(n=3),
                ns.utils.HitsMetric(n=5),
                ns.utils.HitsMetric(n=10)
               ]
    model.compile(optimizer=optimizer,
                    loss=loss,
                    loss_weights = {
                        'concept': 1-args.weight_loss,  
                        'task': args.weight_loss    
                                    },
                    metrics=metrics,
                    run_eagerly=False)

    callbacks = []
    callbacks.append(csv_logger)
    # best_model_callback = MMapModelCheckpoint(model, "val_accuracy", frequency=valid_frequency)
    # callbacks.append(best_model_callback)

    history = model.fit(data_gen_train,
              epochs=args.epochs,
              callbacks=callbacks)
              #validation_data=data_gen_valid,
              #validation_freq=valid_frequency)
    # best_model_callback.restore_weights()

    if output_filename is not None:
        print('Saving model weights to', output_filename)
        model.save_weights(output_filename, overwrite=True)

    print("\nEvaluation train", flush=True)
    train_accuracy = model.evaluate(data_gen_train) 
    print("\nEvaluation val", flush=True)
    # valid_accuracy =  model.evaluate(data_gen_valid) 
    valid_accuracy = 0.0

    # from DataGenerator, generate the next item in data_gen_train, and data_gen_test
    # print('DATASET_TRAIN query', data_gen_train[0][0])
    # print('DATASET_TRAIN label', data_gen_train[0][1])
    # print('DATASET_TEST query', data_gen_test[0][0])
    # print('DATASET_TEST label', data_gen_test[0][1])

    print("\nEvaluation test", flush=True)
    test_accuracy  =  model.evaluate(data_gen_test)
    # test_accuracy  =  model.evaluate(data_gen_train)
 
    print('\nResults',
          '\nTrain', train_accuracy,
          '\nVal', valid_accuracy,
          '\nTest', test_accuracy,
          flush=True)
    print('history:',history.history.keys()) 
    

    # if enable_rules:
    #     # Switching to explain mode
    #     model.explain_mode()

    #     print('\nExplain With Negatives', flush=True)
    #     expls = model.predict(data_gen_train)
    #     for d, expl_layer_output in enumerate(expls):
    #         # [-1] gets the last reasoning depth output.
    #         rules = expl_layer_output[-1]
    #         for r, inputs in rules.items():
    #             print(model.reasoning[d].rules_embedders[r].explain(*inputs))

    #     print('\nExplain Positives', flush=True)
    #     expls = model.predict(data_gen_test_positive_only)
    #     for d, expl_layer_output in enumerate(expls):
    #         # [-1] gets the last reasoning depth output.
    #         rules = expl_layer_output[-1]
    #         for r, inputs in rules.items():
    #             print(model.reasoning[d].rules_embedders[r].explain(*inputs))

    #     # Switching to explain mode
    #     data_gen_test_explain = ns.dataset.DataGenerator(dataset_test, fol, serializer, engine,batch_size=-1, ragged=ragged)
    #     expls = model.predict(data_gen_test_explain)
    #     for d, expl_layer_output in enumerate(expls):
    #         # [-1] gets the last reasoning depth output.
    #         rules = expl_layer_output[-1]
    #         for r, inputs in rules.items():
    #             print(model.reasoning[d].rules_embedders[r].explain(*inputs))

    # return best_model_callback.best_val, 0,  valid_accuracy, test_accuracy, model
    return 0.0, 0.0, valid_accuracy, test_accuracy, model, train_accuracy, history.history
