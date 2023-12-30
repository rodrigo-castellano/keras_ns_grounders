import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import argparse
import keras_ns as ns
from itertools import product
import numpy as np
from os.path import join
import random
import pickle

from dataset import KGCDataHandler
from model import KGCModel
from keras.callbacks import CSVLogger
from keras_ns.logic.commons import Atom, Rule, RuleLoader
from keras_ns.nn.kge import KGEFactory
from keras_ns.utils import MMapModelCheckpoint, KgeLossFactory
from keras_ns.grounding.flat_grounder import DomainGrounder

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

    # Params
    ragged = get_arg(args, 'ragged', None, True)
    valid_frequency = get_arg(args, 'valid_frequency', None, True)

    # Data Loading
    kgc_handler = KGCDataHandler(dataset_name=args.dataset_name,
                                 base_path=base_path,
                                 format=get_arg(args, 'format', None, True),
                                 valid_size=get_arg(args, 'valid_size'))

    dataset_train = kgc_handler.get_dataset(split="train",
                                            number_negatives=args.num_negatives)
    dataset_valid = kgc_handler.get_dataset(split="valid",
                                            number_negatives=args.valid_negatives)
    dataset_test = kgc_handler.get_dataset(split="test")

    fol = kgc_handler.fol


    ### defining rules and grounding engine
    if args.reasoner_depth > 0 and args.num_rules > 0:
        facts = list(fol.facts)
        corruption_atoms = []

        if args.create_flat_rule_list:
            rules = [Rule('r0', body_atoms=[(p.name, "X", "Y")
                                            for p in fol.predicates],
                          head_atoms=[(p.name, "Y", "X")
                                      for p in fol.predicates],
                          var2domain = {"X":fol.domains[0].name, "Y":fol.domains[0].name})]
        else:
            rules = RuleLoader.load(
                join(base_path, args.dataset_name, args.rule_file), args.num_rules)

        engine = DomainGrounder(domains={d.name: d  for d in fol.domains}, rules=rules)
    else:
        rules = []
        engine = None

    serializer = ns.serializer.LogicSerializer(
       predicates=fol.predicates, domains=fol.domains, debug=args.debug)

    # KGE
    kge_embedder = KGEFactory(args.kge)
    assert kge_embedder is not None

    #Loss
    loss_name = get_arg(args, 'loss', 'binary_crossentropy')
    loss = KgeLossFactory(loss_name)

    # The model can be built here or passed from the outside in case of
    # usage of a pre-trained one.
    model = KGCModel(
      fol, rules,
      kge_embedder,
      kge_regularization=args.kge_regularization,
      constant_embedding_size=args.constant_embedding_size,
      kge_atom_embedding_size=args.kge_atom_embedding_size,
      dropout_rate_embedder=args.dropout_rate_embedder,
      reasoner_single_model=get_arg(args, 'reasoner_single_model', False),
      reasoner_atom_embedding_size=args.reasoner_atom_embedding_size,
      reasoner_formula_hidden_embedding_size=args.reasoner_formula_hidden_embedding_size,
      reasoner_regularization=args.reasoner_regularization_factor,
      reasoner_dropout_rate=args.reasoner_dropout_rate,
      reasoner_depth=args.reasoner_depth,
    )


    # Preparing data as generators for model fit
    data_gen_train = ns.dataset.DataGenerator(
        dataset_train, fol, serializer, engine,
        batch_size=args.batch_size, ragged=ragged)

    data_gen_valid = ns.dataset.DataGenerator(
        dataset_valid, fol, serializer, engine,
        batch_size=args.eval_batch_size, ragged=ragged)

    #Building the model (if needed)
    temp_out = model(next(iter(data_gen_train))[0])
    num_outputs = len(temp_out) if isinstance(temp_out,tuple) else 1

    if args.epochs == 0:  # do not train the model, just return it.
        if kge_output_filename:
            print('Reloading kge pretrained model from', kge_output_filename)
            model.kge_model.load_weights(kge_output_filename)
        return model

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)



    metrics = lambda: [ns.utils.MRRMetric(),
               ns.utils.HitsMetric(1),
               ns.utils.HitsMetric(3),
               ns.utils.HitsMetric(5),
               ns.utils.HitsMetric(10)]
    model.compile(optimizer=optimizer,
                  loss=[loss for _ in range(num_outputs)],
                  metrics=[metrics() for _ in range(num_outputs)],
                  run_eagerly=False)

    callbacks = []
    callbacks.append(csv_logger)
    best_model_callback = MMapModelCheckpoint(
        model, "val_mrr" if num_outputs == 1 else "output_1_mrr", maximize=True, frequency=valid_frequency)
    callbacks.append(best_model_callback)

    model.fit(data_gen_train,
                  epochs=args.epochs,
                  callbacks=callbacks,
                  validation_data=data_gen_valid,
                  validation_freq=valid_frequency)
    best_model_callback.restore_weights()

    if output_filename is not None:
        print('Saving model weights to', output_filename)
        model.save_weights(output_filename, overwrite=True)

    if kge_output_filename is not None:
        print('Saving kge model weights to', kge_output_filename)
        model.kge_model.save_weights(kge_output_filename, overwrite=True)

    valid_mrr = 0

    print("Evaluation on test set")
    data_gen_test = ns.dataset.DataGenerator(dataset_test, fol, serializer,
                                             engine, batch_size=args.test_batch_size,
                                             ragged=ragged,
                                             deterministic=True)
    test_scores = model.evaluate(data_gen_test)
    test_mrr = test_scores[1]
    test_hits = test_scores[2:]


    # Switching to explain mode
    model.explain_mode()
    expls = model.predict(data_gen_test)

    for d, expl_layer in enumerate(expls):
        rules = expl_layer[0]
        for r, inputs in rules.items():
            print(model.reasoning[d].rules_embedders[r].explain(*inputs))

    # return best_model_callback.best_val, 0,  valid_accuracy, test_accuracy, model
    return 0, 0,  0, test_mrr, model