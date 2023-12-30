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

from dataset import CollectiveDataHandler
from model import CollectiveModel
from keras.callbacks import CSVLogger
from keras_ns.logic.commons import Atom, Rule, RuleLoader
from keras_ns.nn.kge import KGEFactory
from keras_ns.utils import MMapModelCheckpoint, KgeLossFactory


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
    data_handler = CollectiveDataHandler(dataset_name=args.dataset_name,
                                 base_path=base_path,
                                 format=get_arg(args, 'format', None, True))

    dataset_train = data_handler.get_dataset(split="train")
    dataset_valid = data_handler.get_dataset(split="valid")
    dataset_test = data_handler.get_dataset(split="test")
    fol = data_handler.fol


    ### defining rules and grounding engine
    if args.reasoner_depth > 0 and args.num_rules > 0:
        facts = list(fol.facts)
        # rules = [Rule('r0', body_atoms=[(p.name, "X") for p in fol.predicates if len(p.domains) ==  1 and "_concept" in p.name] + \
        #                                 [(p.name, v[0], v[1]) for v in [("X", "Y"), ("Y", "X")] for p in fol.predicates if
        #                                 len(p.domains) == 2 and "_concept" in p.name],
        #                     head_atoms=[(p.name, "Y") for p in fol.predicates if len(p.domains) == 1 if "_concept" not in p.name],
        #                     var2domain={"X": fol.domains[0].name, "Y": fol.domains[0].name})]
        rules = [Rule('r0', body_atoms = [(p.name, "X") for p in fol.predicates if len(p.domains) ==  1],
                            head_atoms=[(p.name, "Y") for p in fol.predicates if len(p.domains) == 1],
                            var2domain={"X": fol.domains[0].name, "Y": fol.domains[0].name})]
        engine = ns.grounding.ManifoldGrounder(domains={d.name: d for d in fol.domains}, rules=rules, manifold=data_handler.known_facts)
    else:
        rules = []
        engine = None

    serializer = ns.serializer.LogicSerializer(
       predicates=fol.predicates, domains=fol.domains, debug=args.debug)


    # The model can be built here or passed from the outside in case of
    # usage of a pre-trained one.
    model = CollectiveModel(
      fol, rules,
      input_regularization=args.kge_regularization,
      constant_embedding_size=args.constant_embedding_size,
      atom_embedding_size=args.kge_atom_embedding_size,
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

    data_gen_test = ns.dataset.DataGenerator(
        dataset_test, fol, serializer, engine,
        batch_size=args.eval_batch_size, ragged=ragged)

    _ = next(iter(data_gen_train))[0]
    # #Building the model (if needed)
    # model(next(iter(data_gen_train))[0])

    loss = args.loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)


    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics="accuracy",
                  run_eagerly=False)

    callbacks = []
    callbacks.append(csv_logger)
    # best_model_callback = MMapModelCheckpoint(model, "val_accuracy", frequency=valid_frequency)
    # callbacks.append(best_model_callback)

    model.fit(data_gen_train,
                  epochs=args.epochs,
                  callbacks=callbacks,
                  validation_data=data_gen_valid,
                  validation_freq=valid_frequency)
    # best_model_callback.restore_weights()



    if output_filename is not None:
        print('Saving model weights to', output_filename)
        model.save_weights(output_filename, overwrite=True)


    print("Evaluation")
    valid_accuracy = model.evaluate(data_gen_valid)
    test_accuracy = model.evaluate(data_gen_test)


    # Switching to explain mode
    model.explain_mode()
    expls = model.predict(data_gen_test)

    for d, expl_layer in enumerate(expls):
        rules = expl_layer[0]
        for r, inputs in rules.items():
            print(model.reasoning[d].rules_embedders[r].explain(*inputs))

    # return best_model_callback.best_val, 0,  valid_accuracy, test_accuracy, model
    return 0, 0,  valid_accuracy, test_accuracy, model
