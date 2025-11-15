import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import random

# --- Add necessary paths to sys.path ---
# This ensures that the script can find your custom modules like ns_lib, kge_loader, etc.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', 'ns_lib'))

# --- Import your project's modules ---
import ns_lib as ns
from kge_loader import KGCDataHandler
from kge_model import CollectiveModel
from ns_lib.utils import KgeLossFactory, get_arg, load_kge_weights

def set_seeds(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Seeds set to {seed}")

def evaluate(args: argparse.Namespace):
    """
    Main function to load a model and evaluate it on the test set.
    """
    # --- 1. Set Seeds and Load Configuration ---
    set_seeds(args.seed)

    # Construct the path to the configuration file based on the run signature
    name = f"{args.run_signature}_seed_{args.seed}"
    config_filepath = "./config.json"
    # config_filepath = os.path.join(args.ckpt_folder, name, f"{name}_config.json")

    if not os.path.exists(config_filepath):
        raise FileNotFoundError(
            f"Configuration file not found at '{config_filepath}'. "
            "Please ensure you save the config during training."
        )

    print(f"Loading configuration from: {config_filepath}")
    with open(config_filepath, 'r') as f:
        config_dict = json.load(f)
    
    # Convert the loaded dictionary back into an argparse.Namespace for compatibility
    config = argparse.Namespace(**config_dict)
    print("Configuration loaded successfully.")

    # --- 2. Load Data ---
    print("\n--- Loading Data ---")
    data_handler = KGCDataHandler(
        dataset_name=config.dataset_name,
        base_path=args.data_path,
        format=get_arg(config, 'format', 'functional'),
        domain_file=config.domain_file,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        fact_file=config.facts_file
    )
    
    # We only need the test set for evaluation
    dataset_test = data_handler.get_dataset(
        split="test",
        number_negatives=get_arg(config, 'test_negatives', None), # Use all negatives for standard eval
        corrupt_mode=get_arg(config, 'corrupt_mode', 'HEAD_AND_TAIL')
    )
    
    fol = data_handler.fol
    serializer = ns.serializer.LogicSerializerFast(
        predicates=fol.predicates,
        domains=fol.domains,
        constant2domain_name=fol.constant2domain_name
    )

    print("\n--- Creating Test Data Generator ---")
    data_gen_test = ns.dataset.DataGenerator(
        dataset_test,
        fol,
        serializer,
        engine=None, # No grounding engine needed for a pre-trained KGE model
        batch_size=get_arg(config, 'test_batch_size', 128),
        ragged=get_arg(config, 'ragged', True)
    )
    print("Test data generator created.")

    # --- 3. Build Model from Configuration ---
    print("\n--- Building Model from Loaded Configuration ---")
    model = CollectiveModel(
        fol,
        rules=[], # No rules needed for KGE-only evaluation
        kge=config.kge,
        kge_regularization=config.kge_regularization,
        model_name='no_reasoner',
        constant_embedding_size=config.constant_embedding_size,
        predicate_embedding_size=config.predicate_embedding_size,
        kge_atom_embedding_size=config.kge_atom_embedding_size,
        kge_dropout_rate=config.kge_dropout_rate,
        reasoner_depth=0, # Set to 0 as we are only evaluating the KGE part
        resnet=config.resnet,
        # --- Add any other essential parameters from your config ---
        reasoner_single_model=get_arg(config, 'reasoner_single_model', False),
        reasoner_atom_embedding_size=config.reasoner_atom_embedding_size,
        reasoner_formula_hidden_embedding_size=config.reasoner_formula_hidden_embedding_size,
        reasoner_regularization=get_arg(config, 'reasoner_regularization_factor', 0.0),
        reasoner_dropout_rate=config.reasoner_dropout_rate,
        aggregation_type=config.aggregation_type,
        signed=config.signed,
        embedding_resnet=get_arg(config, 'embedding_resnet', False),
        temperature=config.temperature,
        filter_num_heads=config.filter_num_heads,
        filter_activity_regularization=config.filter_activity_regularization,
        num_adaptive_constants=get_arg(config, 'engine_num_adaptive_constants', 0),
        dot_product=get_arg(config, 'engine_dot_product', False),
        cdcr_use_positional_embeddings=get_arg(config, 'cdcr_use_positional_embeddings', True),
        cdcr_num_formulas=get_arg(config, 'cdcr_num_formulas', 3),
        r2n_prediction_type=get_arg(config, 'r2n_prediction_type', 'full'),
        distill=get_arg(config, 'distill', False),
    )
    print("Model built successfully.")

    # --- 4. Compile Model ---
    print("\n--- Compiling Model for Evaluation ---")
    loss = KgeLossFactory(get_arg(config, 'loss', 'binary_crossentropy'))
    metrics = {
        'concept': [ns.utils.MRRMetric(), ns.utils.HitsMetric(1), ns.utils.HitsMetric(10)],
        'task': [ns.utils.MRRMetric(), ns.utils.HitsMetric(1), ns.utils.HitsMetric(10)]
    }
    # model.compile(optimizer='adam', loss={'concept': loss, 'task': loss}, metrics=metrics, run_eagerly=True)
    model.compile(optimizer='adam', run_eagerly=False)
    print("Model compiled with run_eagerly=True.")

    # --- 5. Load Weights ---
    print("\n--- Loading Model Weights ---")
    # First, build the model's layers by calling it on a batch of data
    _ = model(next(iter(data_gen_test))[0])

    # Construct the path to the KGE model weights
    ckpt_filepath = os.path.join(args.ckpt_folder, name, f"{name}_kge_model")
    
    success = load_kge_weights(model, ckpt_filepath, verbose=True)
    if not success:
        raise FileNotFoundError(f"Could not load weights from '{ckpt_filepath}.weights.h5'. Please check the path and signature.")
    print("Weights loaded successfully.")

    # --- 6. Manual Evaluation Loop ---
    print("\n--- Starting Manual Evaluation on Test Set ---")
    
    # Initialize metric objects
    concept_mrr = ns.utils.MRRMetric(name='concept_mrr')
    concept_h1 = ns.utils.HitsMetric(1, name='concept_hits@1')
    concept_h10 = ns.utils.HitsMetric(10, name='concept_hits@10')
    task_mrr = ns.utils.MRRMetric(name='task_mrr')
    task_h1 = ns.utils.HitsMetric(1, name='task_hits@1')
    task_h10 = ns.utils.HitsMetric(10, name='task_hits@10')

    # Iterate over the test dataset
    for i, (x_batch, y_batch) in enumerate(data_gen_test):
        if i >= len(data_gen_test): # Prevent infinite loops with some generators
            break
        print(f"Processing batch {i+1}/{len(data_gen_test)}", end='\r')
        
        # Get model predictions for the current batch
        y_pred = model.predict_on_batch(x_batch)
        
        # Update the state of each metric
        concept_mrr.update_state(y_batch['concept'], y_pred['concept'])
        concept_h1.update_state(y_batch['concept'], y_pred['concept'])
        concept_h10.update_state(y_batch['concept'], y_pred['concept'])
        
        task_mrr.update_state(y_batch['task'], y_pred['task'])
        task_h1.update_state(y_batch['task'], y_pred['task'])
        task_h10.update_state(y_batch['task'], y_pred['task'])
        
    print("\nManual evaluation loop finished.")

    # --- 7. Display Results ---
    print("\n" + "="*50)
    print("          Evaluation Complete")
    print("="*50)
    print(f"Run Signature: {args.run_signature}")
    print(f"Seed: {args.seed}")
    print("\nMetrics:")

    # Collect and print results from the metric objects
    results = {
        'concept_mrr': concept_mrr.result().numpy(),
        'concept_hits@1': concept_h1.result().numpy(),
        'concept_hits@10': concept_h10.result().numpy(),
        'task_mrr': task_mrr.result().numpy(),
        'task_hits@1': task_h1.result().numpy(),
        'task_hits@10': task_h10.result().numpy(),
    }
    for key, value in results.items():
        print(f"  {key:<25}: {value:.4f}")
    print("="*50)


if __name__ == '__main__':
    # --- Disable GPU if not needed, helps avoid memory issues on some systems ---
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.set_soft_device_placement(True)

    parser = argparse.ArgumentParser(description="Streamlined KGE Model Evaluation")
    
    # --- Essential arguments to identify the model to evaluate ---
    parser.add_argument('--seed', type=int, default=0, help="Random seed used during training.")
    parser.add_argument('--data_path', type=str, default='data',
                        help="Base path to the data directory (e.g., 'experiments/data').")
    parser.add_argument('--ckpt_folder', type=str, default='./../checkpoints/',
                        help="Directory where model checkpoints and configs are saved.")
    
    # --- Example of how to run from the command line ---
    # python experiments/evaluate_model.py --run_signature "kinship_family-backward_0_1-no_reasoner-complex-True-256-256-4-rules.txt" --seed 0
    
    args = parser.parse_args()
    args.run_signature = 'kinship_family-backward_0_1-no_reasoner-complex-True-256-256-4-rules.txt'
    # Make paths absolute from the script's location
    args.data_path = os.path.join(current_dir, args.data_path)
    
    evaluate(args)
