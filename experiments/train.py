import os
import tensorflow as tf
import ns_lib as ns
from itertools import product
import numpy as np
from os.path import join
import random
import pickle
from typing import List, Tuple, Dict

from dataset import KGCDataHandler, build_domains
from model import CollectiveModel
from keras.callbacks import CSVLogger
from ns_lib.logic.commons import Atom, Domain, FOL, Rule, RuleLoader
from ns_lib.grounding.grounder_factory import BuildGrounder
from ns_lib.utils import MMapModelCheckpoint, KgeLossFactory, get_arg
from ns_lib.grounding.backward_chaining_grounder_nocleanup import BackwardChainingGrounder_nocleanup
from ns_lib.dataset import _from_strings_to_tensors
import time
from model_utils import * 
import wandb
from wandb.integration.keras import WandbCallback
from wandb.integration.keras import WandbMetricsLogger
# from ultra_utils import  Ultra,nested_dict
from ULTRA.ultra.tasks import build_relation_graph
explain_enabled: bool = False

import itertools
import torch

class Dataset_Ultra():
    
        def __init__(self, edge_index=None, edge_type=None, num_relations=None, num_nodes=None, num_edges=None, device=None, target_edge_index=None, target_edge_type=None):
            self.edge_index = edge_index
            self.edge_type = edge_type
            self.num_relations = num_relations
            self.num_nodes = num_nodes
            self.num_edges = num_edges
            self.device = device
            self.target_edge_index = target_edge_index
            self.target_edge_type = target_edge_type
  
def obtain_queries(dataset,data_handler,serializer,engine,ragged,deterministic,global_serialization):
    queries, labels = dataset[:]
    constants_features = dataset.constants_features
    fol = data_handler.fol
    
    ((X_domains_data, A_predicates_data, A_rules_data, Q, (Q_global,A_predicates_triplets,A_predicates_textualized)), y) = _from_strings_to_tensors(
        fol=fol,
        serializer=serializer,
        queries=queries,
        labels=labels,
        engine=engine,
        ragged=ragged,
        constants_features=constants_features,
        deterministic=deterministic,
        global_serialization=global_serialization) 
    Q_global_positive = [q[0] for q in Q_global]
    # print('\nqueries positive', len(queries), [query[0] for query in queries][:20])
    # print('Q_global_positive', len(Q_global_positive), Q_global_positive[:20])
    return X_domains_data, A_predicates_data, Q_global_positive

def get_ultra_datasets(dataset_train, dataset_valid, dataset_test,data_handler,serializer,engine,ragged,deterministic,global_serialization,relation_graph=build_relation_graph):

    # Get the triplets
    X_domain_train, A_pred_train, train_triplets = obtain_queries(dataset_train,data_handler,serializer,engine,ragged,deterministic,global_serialization)
    X_domain_valid, A_pred_valid, valid_triplets = obtain_queries(dataset_valid,data_handler,serializer,engine,ragged,deterministic,global_serialization)
    X_domain_test, A_pred_test, test_triplets = obtain_queries(dataset_test,data_handler,serializer,engine,ragged,deterministic,global_serialization)

    def unique_ordered(triplets):
        return list(dict.fromkeys(tuple(t) for t in triplets))

    train_triplets = unique_ordered(train_triplets)
    valid_triplets = unique_ordered(valid_triplets)
    test_triplets = unique_ordered(test_triplets)

    # get the number of nodes and relations for the train,val,test set. Do it by getting unique the ones in train, val, test
    train_nodes = [X_domain_train[key].numpy().tolist() for key in X_domain_train]
    valid_nodes = [X_domain_valid[key].numpy().tolist() for key in X_domain_valid]
    test_nodes = [X_domain_test[key].numpy().tolist() for key in X_domain_test]

    # Flatten the lists
    train_nodes = set(itertools.chain(*train_nodes))
    valid_nodes = set(itertools.chain(*valid_nodes))
    test_nodes = set(itertools.chain(*test_nodes))
    num_node = len(train_nodes.union(valid_nodes).union(test_nodes) )

    # do the same for the relations
    train_relations = [key for key in A_pred_train]
    valid_relations = [key for key in A_pred_valid]
    test_relations = [key for key in A_pred_test]
    # take the unique number of relations
    unique_relations = list(set(train_relations+valid_relations+test_relations))
    num_relations_no_inv = torch.tensor(len(unique_relations))
    # num_relations_no_inv = len(data_handler.fol.predicates)
    
    train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
    train_target_etypes = torch.tensor([t[2] for t in train_triplets])
    train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
    train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations_no_inv])

    # valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
    valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
    valid_etypes = torch.tensor([t[2] for t in valid_triplets])

    test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
    test_etypes = torch.tensor([t[2] for t in test_triplets])

    train_data = Dataset_Ultra(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                      target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations_no_inv*2)
    train_data.num_edges = train_data.edge_index.shape[1]
    valid_data = Dataset_Ultra(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                      target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations_no_inv*2)
    valid_data.num_edges = valid_data.edge_index.shape[1]
    test_data = Dataset_Ultra(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                      target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations_no_inv*2)
    test_data.num_edges = test_data.edge_index.shape[1]
    
    # edge_index is the sum of all target_edge_index
    edge_index = torch.cat([train_target_edges, valid_edges, test_edges], dim=1)
    edge_type = torch.cat([train_target_etypes, valid_etypes, test_etypes])
    # num_nodes is given by the train set
    num_edges = None # is not defined in Ultra for the general dataset
    device = 'cpu'
    dataset = Dataset_Ultra(edge_index, edge_type, num_relations_no_inv*2, num_node, num_edges, device)
    filtered_data = dataset 

    train_data = build_relation_graph(train_data)
    valid_data = build_relation_graph(valid_data)
    test_data = build_relation_graph(test_data)

    return train_data, valid_data, test_data, filtered_data



def BuildGrounder(args, fol, rules, facts, domain2adaptive_constants):

    if args.grounder == 'full':
        engine = ns.grounding.PlaceholderGeneratorFullGrounder(
                        domains={d.name:d for d in fol.domains},
                        rules=rules,
                        domain2adaptive_constants=domain2adaptive_constants,
                        exclude_symmetric=True,
                        exclude_query=False)
    elif args.grounder == 'domain':
        engine = ns.grounding.DomainFullGrounder(
                        rules, domains={d.name:d for d in fol.domains},
                        domain2adaptive_constants=domain2adaptive_constants)
    elif args.grounder == 'known':
        engine = ns.grounding.KnownBodyGrounder(rules, facts=facts)
    
    elif 'backward' in args.grounder:
        num_steps = int(args.grounder.split('_')[-1])
        prune_backward = True # if ( ('backward' in args.grounder) and ('prune'in args.grounder) ) else False
        if 'noprune' in args.grounder:
            prune_backward = False

        max_unknown_fact_count_last_step = max_unknown_fact_count = 1
        if 'unknown1' in args.grounder:
            max_unknown_fact_count_last_step = max_unknown_fact_count = 1
        elif 'unknown2' in args.grounder:
            max_unknown_fact_count_last_step = max_unknown_fact_count = 2
        elif 'unknown3' in args.grounder:
            max_unknown_fact_count_last_step = max_unknown_fact_count = 3
        elif 'unknown0' in args.grounder:
            max_unknown_fact_count_last_step = max_unknown_fact_count = 0

        print('Grounder: ',args.grounder,'Number of steps:', num_steps, 'Prune:', prune_backward, 'max_unknown_fact_count:', max_unknown_fact_count)
        engine = ns.grounding.ApproximateBackwardChainingGrounder(
                        rules, facts=facts, domains={d.name:d for d in fol.domains},
                        domain2adaptive_constants=domain2adaptive_constants,
                        pure_adaptive=get_arg(args, 'engine_pure_adaptive', False),
                        num_steps=num_steps,
                        max_unknown_fact_count=max_unknown_fact_count,
                        max_groundings_per_rule=get_arg(
                            args, 'backward_chaining_max_groundings_per_rule', -1),
                        prune_incomplete_proofs=prune_backward)

        if 'original' in args.grounder:
            engine = ns.grounding.BackwardChainingGrounder(
                        rules, facts=facts,
                        domains={d.name:d for d in fol.domains},
                        domain2adaptive_constants=domain2adaptive_constants,
                        pure_adaptive=get_arg(args, 'engine_pure_adaptive', False),
                        num_steps=get_arg(args, 'backward_chaining_depth', 1)) 

        if  'relationentity' in args.grounder:
            engine = ns.grounding.RelationEntityGraphGrounder(
            rules, facts=facts,
            # TODO: Domain support is not added yet.
            #domains={d.name:d for d in fol.domains},
            #domain2adaptive_constants=domain2adaptive_constants,
            build_cartesian_product=True,
            max_elements=get_arg(
                args, 'relation_entity_grounder_max_elements', -1))
        

            
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

 
def main(base_path, output_filename, log_filename, use_WB, args):

    print('\nARGS', args,'\n')
    seed = get_arg(args, 'seed_run_i', 0)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    ragged = get_arg(args, 'ragged', None, True)
    start_train = time.time()

    # DATASET PREPARATION
    data_handler = KGCDataHandler(
        dataset_name=args.dataset_name,
        base_path=base_path,
        format=get_arg(args, 'format', None, True),
        domain_file= args.domain_file,
        train_file= args.train_file,
        valid_file=args.valid_file,
        test_file= args.test_file,
        fact_file= args.facts_file)
    
    dataset_train = data_handler.get_dataset(split="train",number_negatives=args.num_negatives)
    dataset_valid = data_handler.get_dataset(split="valid",number_negatives=args.valid_negatives, corrupt_mode=args.corrupt_mode)
    dataset_test = data_handler.get_dataset(split="test",  number_negatives=args.test_negatives,  corrupt_mode=args.corrupt_mode)
    if explain_enabled and enable_rules and (args.model_name == 'dcr' or args.model_name == 'cdcr'):
        dataset_test_positive_only = data_handler.get_dataset(split="test", number_negatives=0, corrupt_mode=args.corrupt_mode)

    fol = data_handler.fol
    domain2adaptive_constants: Dict[str, List[str]] = None
    dot_product = get_arg(args, 'engine_dot_product', False)

    num_adaptive_constants = get_arg(args, 'engine_num_adaptive_constants', 0)

    # DEFINING RULES AND GROUNDING ENGINE
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

    if args.use_ultra or args.use_ultra_with_kge:
            train_ultra, valid_ultra, test_ultra, _ = get_ultra_datasets(dataset_train, dataset_valid, dataset_test,data_handler,serializer,engine,ragged=True,
                                                                                           deterministic=True,global_serialization=args.global_serialization)
    else:
        train_ultra, valid_ultra, test_ultra, _ = None, None, None, None
    # DATA GENERATORS
    print('***********Generating train data**************')
    start = time.time()
    data_gen_train = ns.dataset.DataGenerator(
        dataset_train, fol, serializer, engine,
        batch_size=args.batch_size, ragged=ragged,
        use_ultra=args.use_ultra, use_ultra_with_kge=args.use_ultra_with_kge, dataset_ultra=train_ultra, use_llm=args.use_llm)
    end = time.time()
    args.time_ground_train = np.round(end - start,2)
    print("Time to create data generator train: ", np.round(end - start,2),'\n************************************')

    start = time.time()
    data_gen_valid = ns.dataset.DataGenerator(
       dataset_valid, fol, serializer, engine,
       batch_size=args.val_batch_size, ragged=ragged,
        use_ultra=args.use_ultra, use_ultra_with_kge=args.use_ultra_with_kge, dataset_ultra=valid_ultra, use_llm=args.use_llm)
    end = time.time()
    args.time_ground_valid = np.round(end - start,2)
    print("Time to create data generator valid: ",  np.round(end - start,2),'\n************************************') 

    start = time.time()
    data_gen_test = ns.dataset.DataGenerator(
        dataset_test, fol, serializer, engine,
        batch_size=args.test_batch_size, ragged=ragged,
        use_ultra=args.use_ultra, use_ultra_with_kge=args.use_ultra_with_kge, dataset_ultra=test_ultra, use_llm=args.use_llm)
    end = time.time()
    args.time_ground_test = np.round(end- start,2)
    print("Time to create data generator test: ",  np.round(end - start,2),'\n************************************')

    print('\nRUNNING TEST SET')
    print('*********************')
    data_gen_train.__getitem__(0)
    print('*********************')


    # COMPILING MODEL
    model = CollectiveModel(
        fol, rules,
        use_ultra=args.use_ultra,
        use_ultra_with_kge=args.use_ultra_with_kge,
        use_llm=args.use_llm,
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
        embedding_resnet=get_arg(args, 'embedding_resnet', False),
        temperature=args.temperature,
        filter_num_heads=args.filter_num_heads,
        filter_activity_regularization=args.filter_activity_regularization,
        num_adaptive_constants=num_adaptive_constants,
        dot_product=dot_product,
        cdcr_use_positional_embeddings=get_arg(
            args, 'cdcr_use_positional_embeddings', True),
        cdcr_num_formulas=get_arg(args, 'cdcr_num_formulas', 3),
        r2n_prediction_type=get_arg(args, 'r2n_prediction_type', 'full'),
        device=args.device,
    )

    #LOSS
    loss_name = get_arg(args, 'loss', 'binary_crossentropy')
    loss = KgeLossFactory(loss_name)

    metrics = [ns.utils.MRRMetric(),
               ns.utils.HitsMetric(1),
               ns.utils.HitsMetric(3),
               ns.utils.HitsMetric(10)]
                # ns.utils.AUCPRMetric()]

    optimizer,lr_scheduler = optimizer_scheduler(args.optimizer,args.lr_sched,args.learning_rate)
    model.compile(optimizer=optimizer,
                    loss=loss,
                    loss_weights = {
                        'concept': 1-args.weight_loss,  
                        'task': args.weight_loss    
                                    },
                    metrics=metrics,
                    run_eagerly=False)

    assert get_arg(args, 'checkpoint_load', None) is None or (
        get_arg(args, 'kge_checkpoint_load', None) is None)
    if get_arg(args, 'checkpoint_load', None) is not None:
        checkpoint_load = get_arg(args, 'checkpoint_load', None)
        print('Loading weights from ', checkpoint_load, flush=True)
        _ = model(next(iter(data_gen_train))[0])  # force building the model.
        if os.path.exists(checkpoint_load):
            model.load_weights(checkpoint_load)
            print('Weights loaded from', checkpoint_load, flush=True)
        model.summary()

    # CALLBACKS
    callbacks = []
    if log_filename is not None:
        csv_logger = ns.utils.CustomCSVLogger(log_filename, append=True, separator=';') 
        callbacks.append(csv_logger)

    if args.lr_sched=='plateau':
      callbacks += [lr_scheduler]
    
    if args.early_stopping:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            patience=40,
            verbose=1)
        callbacks.append(early_stopping)

    best_model_callback = MMapModelCheckpoint(
        model, 'val_task_mrr',
        frequency=args.valid_frequency,
        # if path is not None, checkpoint to file.
        filepath=get_arg(args, 'ckpt_filepath', None))
    callbacks.append(best_model_callback)

    
    
    if not args.use_ultra and not args.use_llm:
        kge_filepath = get_arg(args, 'ckpt_filepath', None)
        if kge_filepath is not None:
            kge_filepath =  '%s_kge_model' % kge_filepath
        kge_best_model_callback = MMapModelCheckpoint(
            model.kge_model, 'val_concept_mrr',
            frequency=args.valid_frequency,
            # if path is not None, checkpoint to file.
            filepath=kge_filepath)
        callbacks.append(kge_best_model_callback)


    # Initialize a W&B run
    if use_WB:
        run = wandb.init(project = "LLM-as-Embedder", name=args.run_signature, config = dict(
                shuffle_buffer = 1024,
                batch_size = args.batch_size,
                learning_rate = args.learning_rate,
                epochs = args.epochs)) 
        callbacks.append(WandbMetricsLogger(log_freq=10))


    if args.epochs > 0:

        # TRAIN
        history = model.fit(data_gen_train,
                epochs=args.epochs,
                callbacks=callbacks,
                validation_data=data_gen_valid,
                validation_freq=args.valid_frequency)
        
        end_train = time.time()

        # Close the W&B run
        if use_WB:
            run.finish()
        
        args.time_train = np.round(end_train - start_train,2)
        print('Training time:', np.round(end_train - start_train,2), 'seconds')
        best_model_callback.restore_weights()

        if output_filename is not None:
            print('Saving model weights to', output_filename)
            model.save_weights(output_filename, overwrite=True)
    # else:
    #     history = None
    #     best_model_callback._weights_saved = True
    #     best_model_callback._last_checkpoint_filename = args.file_name_saved_weights
    #     best_model_callback.restore_weights()

    # EVALUATION
    print("\nEvaluation train", flush=True)
    model.test_mode('train',mode=True)
    train_accuracy = model.evaluate(data_gen_train)#,train_data=True,testing=True) 
    print("\nEvaluation val", flush=True)
    model.test_mode('valid',mode=True)
    valid_accuracy =  model.evaluate(data_gen_valid)#,val_data=True,testing=True) 
    print("\nEvaluation test", flush=True)
    start_inf = time.time()
    model.test_mode('test',mode=True)
    test_accuracy  =  model.evaluate(data_gen_test)#,test_data=True,testing=True)
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