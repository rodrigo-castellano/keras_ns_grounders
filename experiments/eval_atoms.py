import os
import argparse
import tensorflow as tf
import numpy as np
import random
from os.path import join
import sys

# Add relevant paths to the system path to import necessary modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', 'ns_lib'))

import ns_lib as ns
from dataset import KGCDataHandler
from model import CollectiveModel
from ns_lib.utils import load_kge_weights
from ns_lib.logic.commons import Atom, Predicate

def evaluate_single_atom(model, data_handler, serializer, atom_string):
    """
    Evaluates a single atom using the provided pre-trained model.

    Args:
        model (CollectiveModel): The pre-trained KGE model.
        data_handler (KGCDataHandler): The data handler with dataset information.
        serializer (LogicSerializerFast): The serializer for converting atoms to indices.
        atom_string (str): The atom to evaluate in string format (e.g., "locatedIn(spain,europe)").

    Returns:
        float: The prediction score for the atom.
    """
    fol = data_handler.fol

    # 1. Parse the atom string into a tuple representation
    atom_to_evaluate = Atom(s=atom_string, format="functional").toTuple()
    print(f"Evaluating atom: {atom_to_evaluate}")

    # 2. Use the serializer to convert the atom to the model's input format.
    queries = [[atom_to_evaluate]]
    labels = [[1.0]]  # Dummy label

    engine = None  # No reasoning engine needed
    (
        (X_domains, A_predicates, A_rules, Q),
        y,
    ) = ns.dataset._from_strings_to_tensors(
        fol=fol,
        serializer=serializer,
        queries=queries,
        labels=labels,
        engine=engine,
        ragged=False,
    )

    # 3. Make a prediction by calling the KGE sub-model's `call` method directly.
    # Using .call() instead of .predict() bypasses the strict input cardinality checks
    # that cause errors when evaluating a single atom.
    atom_outputs, _ = model.kge_model.call((X_domains, A_predicates))
    
    # The score is the first (and only) element in the output tensor.
    score = atom_outputs[0][0]

    return score

def main():
    # --- Configuration ---
    # Create a namespace object to hold the configuration parameters
    args = argparse.Namespace()

    # Parameters from the runner.py log to ensure exact replication
    args.aggregation_type = 'max'
    args.batch_size = 256
    args.cdcr_num_formulas = 3
    args.cdcr_use_positional_embeddings = False
    args.ckpt_folder = './../checkpoints/'
    args.constant_embedding_size = 200
    args.corrupt_mode = 'TAIL'
    args.data_path = 'experiments/data'
    args.dataset_name = 'countries_s3'
    args.distill = False
    args.distill_kge_labels = False
    args.domain_file = 'domain2constants.txt'
    args.dropout = 0.0
    args.dropout_rate_embedder = 0.0
    args.early_stopping = True
    args.engine_num_adaptive_constants = 0
    args.engine_num_negatives = None
    args.epochs = 100
    args.facts_file = 'facts.txt'
    args.filter_activity_regularization = 0.0
    args.filter_num_heads = 3
    args.format = 'functional'
    args.grounder = 'backward_0_1'
    args.kge = 'complex'
    args.kge_atom_embedding_size = 100
    args.kge_dropout_rate = 0.0
    args.kge_regularization = 0.0
    args.learning_rate = 0.01
    args.load_kge_ckpt = True
    args.load_model_ckpt = False
    args.log_folder = './experiments/runs/'
    args.loss = 'binary_crossentropy'
    args.lr_sched = 'plateau'
    args.model_name = 'no_reasoner'
    args.num_negatives = 1
    args.num_rules = 0
    args.optimizer = 'adam'
    args.predicate_embedding_size = 200
    args.r2n_prediction_type = 'full'
    args.ragged = True
    args.reasoner_atom_embedding_size = 100
    args.reasoner_depth = 0 # Set to 0 for KGE-only evaluation
    args.reasoner_dropout_rate = 0.0
    args.reasoner_formula_hidden_embedding_size = 100
    args.reasoner_regularization_factor = 0.0
    args.reasoner_single_model=False # Added for completeness
    args.resnet = True
    args.rules_file = 'rules.txt'
    args.run_signature = 'countries_s3-backward_0_1-no_reasoner-complex-True-256-256-128-rules.txt'
    args.save_kge_ckpt = True
    args.save_model_ckpt = True
    args.seed = 0
    args.signed = True
    args.stop_kge_gradients = False
    args.store_ranks = False
    args.temperature = 0.0
    args.test_batch_size = 128
    args.test_file = 'test.txt'
    args.test_negatives = None
    args.train_file = 'train.txt'
    args.use_WB = False
    args.use_logger = True
    args.val_batch_size = 256
    args.valid_file = 'valid.txt'
    args.valid_frequency = 1
    args.valid_negatives = 100
    args.valid_size = None
    args.weight_loss = 0.5

    # Atom to evaluate
    # args.atom = 'locatedInCR(burkina_faso,africa)'
    args.atom = 'locatedInCR(italy,europe)'
    
    # --- End Configuration ---

    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # 1. Load data and create mappings
    print("Loading data...")
    data_handler = KGCDataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        format=args.format,
        domain_file=args.domain_file,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        fact_file=args.facts_file
    )

    fol = data_handler.fol
    
    # Manually add the predicate of the atom to be evaluated if it's not already in the FOL vocabulary.
    atom_to_parse = Atom(s=args.atom, format="functional")
    predicate_name = atom_to_parse.r
    if predicate_name not in fol.name2predicate:
        print(f"Predicate '{predicate_name}' not in vocabulary. Adding it manually.")
        name2domain = {d.name: d for d in fol.domains}
        try:
            constant_domains = [name2domain[data_handler.constant2domain[c]] for c in atom_to_parse.args]
            new_predicate = Predicate(predicate_name, constant_domains)
            fol.predicates.append(new_predicate)
            fol.name2predicate[predicate_name] = new_predicate
            fol.name2predicate_idx[predicate_name] = len(fol.predicates) - 1
            print(f"Successfully added predicate: {new_predicate}")
        except KeyError as e:
            print(f"Error: A constant '{e.args[0]}' in your atom is not in any known domain.")
            return

    serializer = ns.serializer.LogicSerializerFast(
        predicates=fol.predicates,
        domains=fol.domains,
        constant2domain_name=fol.constant2domain_name
    )
    print("Data loaded and FOL object updated.")
    
    # 2. Build the model architecture
    print("Building model...")
    model = CollectiveModel(
        fol,
        rules=[],
        kge=args.kge,
        kge_regularization=args.kge_regularization,
        constant_embedding_size=args.constant_embedding_size,
        predicate_embedding_size=args.predicate_embedding_size,
        kge_atom_embedding_size=args.kge_atom_embedding_size,
        kge_dropout_rate=args.kge_dropout_rate,
        reasoner_depth=args.reasoner_depth,
        model_name=args.model_name,
        reasoner_atom_embedding_size=args.reasoner_atom_embedding_size,
        reasoner_formula_hidden_embedding_size=args.reasoner_formula_hidden_embedding_size,
        reasoner_regularization=args.reasoner_regularization_factor,
        reasoner_single_model=args.reasoner_single_model,
        reasoner_dropout_rate=args.reasoner_dropout_rate,
        aggregation_type=args.aggregation_type,
        signed=args.signed,
        temperature=args.temperature,
        resnet=args.resnet,
        embedding_resnet=False,
        filter_num_heads=args.filter_num_heads,
        filter_activity_regularization=args.filter_activity_regularization,
        num_adaptive_constants=args.engine_num_adaptive_constants,
        dot_product=False,
        cdcr_use_positional_embeddings=args.cdcr_use_positional_embeddings,
        cdcr_num_formulas=args.cdcr_num_formulas,
        r2n_prediction_type=args.r2n_prediction_type,
        distill=args.distill,
    )

    # Build the model by passing a dummy input.
    print("Creating dummy input to build model...")
    dummy_dataset = data_handler.get_dataset(split="test", number_negatives=0)
    dummy_generator = ns.dataset.DataGenerator(
        dummy_dataset, fol, serializer, engine=None, batch_size=1, ragged=False
    )
    dummy_input = next(iter(dummy_generator))[0]
    
    model(dummy_input)
    model.kge_model((dummy_input[0], dummy_input[1]))
    
    print("Model built.")

    # 3. Load pre-trained weights
    print("Loading pre-trained weights...")
    name = args.run_signature + '_seed_' + str(args.seed)
    ckpt_filepath = os.path.join(args.ckpt_folder, name, name + '_kge_model')
    
    print(f"Attempting to load weights from: {ckpt_filepath}.weights.h5")
    
    if not load_kge_weights(model, ckpt_filepath, verbose=True):
        print(f"Could not load weights from {ckpt_filepath}. Exiting.")
        return

    # 4. Evaluate the single atom
    score = evaluate_single_atom(model, data_handler, serializer, args.atom)

    print(f"\nPrediction score for atom '{args.atom}': {score:.4f}")

if __name__ == '__main__':
    main()
