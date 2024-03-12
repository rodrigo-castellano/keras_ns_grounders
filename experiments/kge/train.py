import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import keras_ns as ns
from itertools import product
import numpy as np
from os.path import join
import random
import pickle
from typing import List, Tuple

from dataset import KGCDataHandler, build_domains
from model import CollectiveModel
from keras.callbacks import CSVLogger
from keras_ns.logic.commons import Atom, Domain, FOL, Rule, RuleLoader
from keras_ns.utils import MMapModelCheckpoint, KgeLossFactory, read_file_as_lines
from keras_ns.utils import get_arg
from keras_ns.grounding.backward_chaining_grounder_nocleanup import BackwardChainingGrounder_nocleanup
from typing import Dict, List
import time

explain_enabled: bool = False

def BuildGrounder(args, fol, rules, facts, domain2adaptive_constants):

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
        engine = ns.grounding.KnownBodyGrounder(rules, facts=facts)
    
    elif 'backward' in args.grounder:
        num_steps = int(args.grounder.split('_')[-1])
        prune_backward = True # if ( ('backward' in args.grounder) and ('prune'in args.grounder) ) else False
        print('Grounder: ',args.grounder,'Number of steps:', num_steps, 'Prune:', prune_backward)
        engine = ns.grounding.BackwardChainingGrounder(rules, facts=facts,
                                                    domains={d.name:d for d in fol.domains},
                                                    num_steps=num_steps, prune_incomplete_proofs=prune_backward)
        if '_nocleanup' in args.grounder:
            engine = BackwardChainingGrounder_nocleanup(rules, facts=facts,
                                                    domains={d.name:d for d in fol.domains},
                                                    num_steps=num_steps, prune_incomplete_proofs=prune_backward)
    elif args.grounder == 'domainbody':
        engine = ns.grounding.DomainBodyGrounder(domains={d.name:d for d in fol.domains},
                                                rules=rules,
                                                exclude_symmetric=True,
                                                exclude_query=False)
    elif args.grounder == 'relationentity':
        engine = ns.grounding.RelationEntityGraphGrounder(
            rules, facts=facts,
            build_cartesian_product=True,
            max_elements=20)
    return engine
 
def main(base_path, output_filename, kge_output_filename, log_filename, args):

    csv_logger = ns.utils.CustomCSVLogger(log_filename, append=True, separator=';') 
    print('\nARGS', args,'\n')

    seed = get_arg(args, 'seed_run_i', 0)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Params
    ragged = get_arg(args, 'ragged', None, True)

    start_train = time.time()

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
    dataset_valid = data_handler.get_dataset(
        split="valid",
       number_negatives=args.valid_negatives, corrupt_mode=args.corrupt_mode)
    
    if explain_enabled:
        dataset_test_positive_only = data_handler.get_dataset(
            split="test", number_negatives=0, corrupt_mode=args.corrupt_mode)

    fol = data_handler.fol
    domain2adaptive_constants: Dict[str, List[str]] = None
    num_adaptive_constants = get_arg(args, 'engine_num_adaptive_constants', 0)

    # Domains are used up to the serializer, the model assumes that all
    # constants are in the same domain.

    ### defining rules and grounding engine
    rules = []
    engine = None

    enable_rules = (args.reasoner_depth > 0 and args.num_rules > 0)
    if enable_rules:
        rules = ns.utils.read_rules(join(base_path, args.dataset_name, args.rules_file),args)
        facts = list(data_handler.train_known_facts_set)
        engine = BuildGrounder(args, fol, rules, facts, domain2adaptive_constants)

    serializer = ns.serializer.LogicSerializerFast(
        predicates=fol.predicates, domains=fol.domains,
        constant2domain_name=fol.constant2domain_name,
        domain2adaptive_constants=domain2adaptive_constants)

    # Preparing data as generators for model fit
    print('Generating train data')
    start = time.time()
    data_gen_train = ns.dataset.DataGenerator(
        dataset_train, fol, serializer, engine,
        batch_size=args.batch_size, ragged=ragged)
    end = time.time()
    args.time_ground_train = np.round(end - start,2)
    print("Time to create data generator train: ", np.round(end - start,2))

    start = time.time()
    data_gen_valid = ns.dataset.DataGenerator(
       dataset_valid, fol, serializer, engine,
       batch_size=args.val_batch_size, ragged=ragged)
    end = time.time()
    args.time_ground_valid = np.round(end - start,2)
    print("Time to create data generator valid: ",  np.round(end - start,2))

    # The model can be built here or passed from the outside in case of
    # usage of a pre-trained one.
    model = CollectiveModel(
        fol, rules,
        kge=args.kge,
        kge_regularization=args.kge_regularization,
        model_name=get_arg(args, 'model_name', 'dcr'),
        constant_embedding_size=args.constant_embedding_size,
        predicate_embedding_size=args.predicate_embedding_size,
        kge_atom_embedding_size=args.kge_atom_embedding_size,
        kge_dropout_rate=args.kge_dropout_rate,
        reasoner_single_model=get_arg(args, 'reasoner_single_model', False),
        reasoner_atom_embedding_size=args.reasoner_atom_embedding_size,
        reasoner_formula_hidden_embedding_size=args.reasoner_formula_hidden_embedding_size,
        reasoner_regularization=args.reasoner_regularization_factor,
        reasoner_dropout_rate=args.reasoner_dropout_rate,
        reasoner_depth=args.reasoner_depth,
        aggregation_type=args.aggregation_type,
        signed=args.signed,
        resnet=get_arg(args, 'resnet', False),
        temperature=args.temperature,
        filter_num_heads=args.filter_num_heads,
        filter_activity_regularization=args.filter_activity_regularization,
        num_adaptive_constants=num_adaptive_constants,
        cdcr_use_positional_embeddings=get_arg(
            args, 'cdcr_use_positional_embeddings', True),
        cdcr_num_formulas=get_arg(args, 'cdcr_num_formulas', 3),
    )

    #Loss
    loss_name = get_arg(args, 'loss', 'binary_crossentropy')
    loss = KgeLossFactory(loss_name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    metrics = [ns.utils.MRRMetric(),
               ns.utils.HitsMetric(1),
               ns.utils.HitsMetric(3),
               ns.utils.HitsMetric(10)]
    model.compile(optimizer=optimizer,
                    loss=loss,
                    loss_weights = {
                        'concept': 1-args.weight_loss,  
                        'task': args.weight_loss    
                                    },
                    metrics=metrics,
                    run_eagerly=False)

    if get_arg(args, 'kge_checkpoint_load', None) is not None:
        kge_checkpoint_load = get_arg(args, 'kge_checkpoint_load', None)
        print('Loading weights from ', kge_checkpoint_load[0], 'Trainable', kge_checkpoint_load[1], flush=True)
        _ = model(next(iter(data_gen_train))[0])  # force building the model.
        print('Preload model', flush=True)
        model.kge_model.summary()
        model.kge_model.load_weights(kge_checkpoint_load[0])
        model.kge_model.trainable = kge_checkpoint_load[1]
        model.summary()

    callbacks = []
    callbacks.append(csv_logger)

    best_model_callback = MMapModelCheckpoint(model, 'val_task_mrr',frequency=args.valid_frequency,
        # if path is not None, checkpoint to file.
        filepath=get_arg(args, 'ckpt_filepath', None))        
    callbacks.append(best_model_callback)

    # kge_filepath = get_arg(args, 'ckpt_filepath', None)
    # if kge_filepath is not None:
    #     kge_filepath = os.path.join(kge_filepath, 'kge_model')
    # kge_best_model_callback = MMapModelCheckpoint(
    #     model.kge_model, 'val_concept_mrr',
    #     frequency=args.valid_frequency,
    #     # if path is not None, checkpoint to file.
    #     filepath=kge_filepath)
    # callbacks.append(kge_best_model_callback)

    history = model.fit(data_gen_train,
              epochs=args.epochs,
              callbacks=callbacks,
              validation_data=data_gen_valid,
              validation_freq=args.valid_frequency
              )
    
    end_train = time.time()
    args.time_train = np.round(end_train - start_train,2)
    print('Training time:', np.round(end_train - start_train,2), 'seconds')
    best_model_callback.restore_weights()

    if output_filename is not None:
        print('Saving model weights to', output_filename)
        model.save_weights(output_filename, overwrite=True)

    print("\nEvaluation train", flush=True)
    train_accuracy = model.evaluate(data_gen_train) 
    print("\nEvaluation val", flush=True)
    valid_accuracy =  model.evaluate(data_gen_valid) 
    # valid_accuracy = list(np.zeros(len(train_accuracy)))

    print("\nEvaluation test", flush=True)
    start_inf = time.time()
    dataset_test = data_handler.get_dataset(split="test", corrupt_mode=args.corrupt_mode,number_negatives=args.test_negatives)
    if explain_enabled and enable_rules and (args.model_name == 'dcr' or args.model_name == 'cdcr'):
        dataset_test_positive_only = data_handler.get_dataset(
            split="test", number_negatives=0, corrupt_mode='TAIL')
        
    start = time.time()
    data_gen_test = ns.dataset.DataGenerator(
        dataset_test, fol, serializer, engine,
        batch_size=args.test_batch_size, ragged=ragged)
    end = time.time()
    args.time_ground_test = np.round(end- start,2)
    print("Time to create data generator test: ",  np.round(end - start,2))
    test_accuracy  =  model.evaluate(data_gen_test)
    end_inf = time.time()
    args.time_inference = np.round(end_inf - start_inf,2)
    print('Inference time:', np.round(end_inf - start_inf,2), 'seconds')

    print('Metrics,loss:',history.history.keys()) 
    print('\nResults',
          '\nTrain', np.round(train_accuracy,3),
          '\nVal', np.round(valid_accuracy,3),
          '\nTest', np.round(test_accuracy,3),
          flush=True)

    if explain_enabled and enable_rules and (args.model_name == 'dcr' or args.model_name == 'cdcr'):
        model.explain_mode(True)
        print('\nExplain Train', flush=True)
        print(model.predict(data_gen_train)[-1])

        print('\nExplain Test', flush=True)
        data_gen_test_explain = ns.dataset.DataGenerator(
            dataset_test, fol, serializer, engine, batch_size=-1, ragged=ragged)
        print(model.predict(data_gen_test_explain)[-1])

        data_gen_test_positive_only = ns.dataset.DataGenerator(
            dataset_test_positive_only, fol, serializer, engine,
            batch_size=args.test_batch_size, ragged=ragged)
        for r in model.reasoning[-1].rule_embedders.values():
            r._verbose=True
        print(model.predict(data_gen_test_positive_only)[-1])

    return train_accuracy,valid_accuracy, test_accuracy, history.history
