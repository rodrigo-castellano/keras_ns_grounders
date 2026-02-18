import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', 'ns_lib'))

import random
from os.path import join
from typing import Dict, List, Tuple, Union, Iterable
import numpy as np
import tensorflow as tf
import ns_lib as ns
from ns_lib.grounding import *
from model_utils import *
from ns_lib.logic.commons import Atom, FOL,RuleGroundings
from dataset import KGCDataHandler
import argparse
from ns_lib.grounding.grounder_factory import BuildGrounder
from ns_lib.grounding.backward_chaining_grounder import ApproximateBackwardChainingGrounder
from ns_lib.grounding.backward_chaining_exact import BackwardChainingGrounder


'''
This script is used to test the grounding of rules in a knowledge graph.
Mainly used to find the number of groundings for each rule.
'''

parser = argparse.ArgumentParser(description='Test grounding of rules in a knowledge graph')
parser.add_argument('-d', '--dataset', type=str, default='kinship_family',
                    help='Dataset name (default: deep_chain_v4)')
parser.add_argument('-g', '--grounder', type=str, default='backward_1_1',
                    help='Grounder type, e.g. backward_1, backward_1_1, backward_1_2 (default: backward_1_1)')
parser.add_argument('-p', '--data_path', type=str, default='experiments/data',
                    help='Base data path (default: experiments/data)')
parser.add_argument('-r', '--rules_file', type=str, default='rules.txt',
                    help='Rules file name (default: rules.txt)')
parser.add_argument('--print_groundings', action='store_true',
                    help='Print ground formulas for each query')
parser.add_argument('-s', '--split', type=str, default='test',
                    help='Data split to use: train, valid, test (default: test)')
args = parser.parse_args()

args.dataset_name = args.dataset
args.facts_file = 'facts.txt'
args.train_file = 'train.txt'
args.valid_file = 'valid.txt'
args.test_file = 'test.txt'
args.domain_file = 'domain2constants.txt'
args.num_negatives = 0
args.test_negatives = 0
args.format = "functional"

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Data Loading
data_handler = KGCDataHandler(
    dataset_name=args.dataset_name,
    base_path=args.data_path,
    format=args.format,
    domain_file= args.domain_file,
    train_file= args.train_file,
    valid_file=args.valid_file,
    test_file= args.test_file,
    fact_file= args.facts_file)


dataset = data_handler.get_dataset(split=args.split, number_negatives=args.num_negatives)

fol = data_handler.fol
facts = fol.facts
rules = ns.utils.read_rules(join(args.data_path, args.dataset_name, args.rules_file),args)
queries, labels = dataset[0:len(dataset)]
print(f'Split: {args.split}, number of queries: {len(queries)}')


type = args.grounder
print('Building Grounder', type, flush=True)

if 'backward' in type:
    # if the count of '_' the name is 2, it means that the parameter 'a' is included. Else there is no parameter a. It goes after the first '_'
    backward_width = None
    if type.count('_') == 2:
        backward_width = int(type[type.index('_')+1]) # take the first character after the first '_'
        backward_depth = int(type[-1])
        type = 'ApproximateBackwardChainingGrounder'
    else:
        backward_depth = int(type[-1])
        type = 'BackwardChainingGrounder'
    prune_incomplete_proofs = False if 'noprune' in args.grounder else True
    print('Grounder: ',args.grounder,'backward_depth:', backward_depth, 'Prune:', prune_incomplete_proofs, 'backward_width:', backward_width)

if type == 'ApproximateBackwardChainingGrounder':
    # Requires Horn Clauses.
    engine = ApproximateBackwardChainingGrounder(
        rules, facts=facts, domains={d.name:d for d in fol.domains},
        domain2adaptive_constants=None,
        pure_adaptive=get_arg(args, 'engine_pure_adaptive', False),
        num_steps=backward_depth,
        max_unknown_fact_count=backward_width,
        max_unknown_fact_count_last_step=backward_width,
        prune_incomplete_proofs=prune_incomplete_proofs,
        max_groundings_per_rule=get_arg(
            args, 'backward_chaining_max_groundings_per_rule', -1),
        force_determinism=False)

elif type == 'BackwardChainingGrounder':
    # Requires Horn Clauses.
    engine = BackwardChainingGrounder(
        rules, facts=facts, domains={d.name:d for d in fol.domains},
        domain2adaptive_constants=None,
        pure_adaptive=get_arg(args, 'engine_pure_adaptive', False),
        num_steps=backward_depth,
        accumulate_groundings=False,
        )



# engine = BuildGrounder(args, rules, facts=facts, fol=fol, domain2adaptive_constants=None)

import time
start = time.time()

queries = queries[:]
print('number of queries:',len(queries))

len_groundings = []
n_queries_with_groundings = 0
queries_with_groundings = []
for i,query in enumerate(queries):
    facts = sorted(facts)
    ground_formulas = engine.ground(sorted(tuple(facts)),tuple(ns.utils.to_flat(query)),deterministic=True)

    n_ground = len([grounding for rule in ground_formulas for grounding in ground_formulas[rule]])
    print('num groundings:', n_ground)
    if args.print_groundings:
        print('ground_formulas:', ground_formulas)
    len_groundings.append(n_ground)
    n_queries_with_groundings += 1 if len_groundings[-1] > 0 else 0
    print('n_queries_with_groundings:',n_queries_with_groundings)
    predicate = query[0][0]
    cte1 = query[0][1]
    cte2 = query[0][2]
    query_str = f"{predicate}({cte1},{cte2})."
    queries_with_groundings.append(query_str) if len_groundings[-1] > 0 and query_str not in queries_with_groundings else 0

print('num groundings:', sum(len_groundings),'avg number of grounding:',round(np.mean(len_groundings),3), 'std:',round(np.std(len_groundings),3))
print('coverage (of groundings, not rules):',round(n_queries_with_groundings/len(queries),3))

print('Time:',round(time.time()-start,3))

# --- Batched grounding: all queries at once (exposes cascading proof propagation) ---
print('\n' + '='*80)
print('BATCHED GROUNDING (all queries at once)')
print('='*80)

# Re-create engine (fresh state)
if type == 'ApproximateBackwardChainingGrounder':
    engine_batch = ApproximateBackwardChainingGrounder(
        rules, facts=list(fol.facts), domains={d.name:d for d in fol.domains},
        domain2adaptive_constants=None,
        pure_adaptive=get_arg(args, 'engine_pure_adaptive', False),
        num_steps=backward_depth,
        max_unknown_fact_count=backward_width,
        max_unknown_fact_count_last_step=backward_width,
        prune_incomplete_proofs=prune_incomplete_proofs,
        max_groundings_per_rule=get_arg(
            args, 'backward_chaining_max_groundings_per_rule', -1),
        force_determinism=False)
elif type == 'BackwardChainingGrounder':
    engine_batch = BackwardChainingGrounder(
        rules, facts=list(fol.facts), domains={d.name:d for d in fol.domains},
        domain2adaptive_constants=None,
        pure_adaptive=get_arg(args, 'engine_pure_adaptive', False),
        num_steps=backward_depth,
        accumulate_groundings=False)

# Flatten all queries into a single list of tuples
all_flat_queries = []
query_predicates = []
for query in queries:
    flat = tuple(ns.utils.to_flat(query))
    all_flat_queries.extend(flat)
    query_predicates.append(query[0][0])

start_batch = time.time()
ground_formulas_batch = engine_batch.ground(
    sorted(tuple(fol.facts)), tuple(all_flat_queries), deterministic=True)
batch_time = time.time() - start_batch

# Collect all grounded head atoms across all rules
grounded_heads = set()
total_groundings_batch = 0
for rule_name, rg in ground_formulas_batch.items():
    for grounding in rg.groundings:
        head_atoms = grounding[0]  # tuple of head atom tuples
        total_groundings_batch += 1
        for h in head_atoms:
            grounded_heads.add(h)

# Check per-predicate coverage
from collections import Counter
pred_total = Counter()
pred_grounded = Counter()
for query in queries:
    pred = query[0][0]
    q_tuple = tuple(query[0])
    pred_total[pred] += 1
    if q_tuple in grounded_heads:
        pred_grounded[pred] += 1

n_grounded_batch = sum(pred_grounded.values())
print(f'Total queries (batched): {len(all_flat_queries)}, Total groundings: {total_groundings_batch}')
print(f'Coverage (batched): {n_grounded_batch}/{len(queries)} = {round(n_grounded_batch/len(queries),3)}')
print(f'\nPer-predicate coverage:')
for pred in sorted(pred_total.keys()):
    g = pred_grounded.get(pred, 0)
    t = pred_total[pred]
    print(f'  {pred}: {g}/{t} = {round(g/t,3)}')
print(f'Time (batched): {round(batch_time,3)}s')

# # write the queries into a file with the name of the dataset, the grounder
# with open(f'test_{args.dataset_name}_{args.grounder}.txt', 'w') as f:
#     for item in queries_with_groundings:
#         f.write("%s\n" % item)
